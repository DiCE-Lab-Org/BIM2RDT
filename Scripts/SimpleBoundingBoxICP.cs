using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;                    // fixes StringComparer, StringSplitOptions
using System.Globalization;      // CharUnicodeInfo
using System.Text;               // StringBuilder, Normalization
using System.Text.RegularExpressions;

[DisallowMultipleComponent]
public class SimpleBoundingBoxICP : MonoBehaviour
{

    bool biasReady;
    public bool use_bias = true;
private float uprightB = 0.0f;
    // Put inside SimpleBoundingBoxICP class (as nested static helper) or as a separate static class
public static class LabelNorm
{
    static readonly Regex NonAlnum = new Regex(@"[^a-z0-9]+", RegexOptions.Compiled);
    static readonly HashSet<string> Stop = new HashSet<string>{
        "the","a","an","of","and","rw","bim","virtual","detected","real","world"
    };
    // Map loose synonyms to a canonical token
    static readonly Dictionary<string,string> Syn = new Dictionary<string,string>{
        {"wooden","wood"}, {"crate","box"}, {"case","box"}, {"container","box"},
        {"lamp","light"}, {"lights","light"}, {"ceiling","ceiling"}, {"floor","floor"}
    };

    // Cache to reduce allocations
    static readonly Dictionary<string, HashSet<string>> Cache = new Dictionary<string, HashSet<string>>();

    public static string CanonicalKey(string s)
    {
        var toks = Tokens(s);
        if (toks.Count == 0) return "";
        var arr = toks.ToList();
        arr.Sort(StringComparer.Ordinal);
        return string.Concat(arr); // stable key for dictionaries
    }

    public static bool Match(string a, string b, float jaccardThreshold = 0.6f)
    {
        var A = Tokens(a);
        var B = Tokens(b);
        if (A.SetEquals(B)) return true;
        if (A.IsSubsetOf(B) || B.IsSubsetOf(A)) return true;

        int inter = A.Intersect(B).Count();
        if (inter == 0) return false;
        var union = new HashSet<string>(A); union.UnionWith(B);
        float jacc = (float)inter / union.Count;
        return jacc >= jaccardThreshold;
    }

    static HashSet<string> Tokens(string s)
    {
        if (string.IsNullOrWhiteSpace(s)) return new HashSet<string>();
        if (Cache.TryGetValue(s, out var cached)) return cached;

        // 1) lowercase + strip diacritics
        string lower = s.ToLowerInvariant();
        string decomp = lower.Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder(decomp.Length);
        foreach (var ch in decomp)
            if (CharUnicodeInfo.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark)
                sb.Append(ch);
        string ascii = sb.ToString().Normalize(NormalizationForm.FormC);

        // 2) collapse to spaces, split
        string spaced = NonAlnum.Replace(ascii, " ");
        var parts = spaced.Split(new[]{' '}, StringSplitOptions.RemoveEmptyEntries);

        var outSet = new HashSet<string>();
        foreach (var raw in parts)
        {
            string t = raw;
            if (Stop.Contains(t)) continue;

            // synonyms
            if (Syn.TryGetValue(t, out var mapped)) t = mapped;

            // naive singularization
            if (t.EndsWith("ies")) t = t.Substring(0, t.Length - 3) + "y";
            else if (t.EndsWith("es") && t.Length > 3) t = t.Substring(0, t.Length - 2);
            else if (t.EndsWith("s") && t.Length > 2) t = t.Substring(0, t.Length - 1);

            if (t.Length < 2) continue; // drop tiny fragments
            outSet.Add(t);
        }

        Cache[s] = outSet;
        return outSet;
    }
}
    [Header("Core Components")]
    public OpenCVCornersXYZOverlay cornersDetector;  // Your corners/features detector
    public ICPAligner icp;                           // ICP alignment component
    public Camera cam;                               // Main camera
    [Header("YOLO Validation")]
    public DetectionOverlay yoloDetector;           // Reference to your YOLO detection overlay
    [Range(0.1f, 1.0f)] public float minYoloConfidence = 0.35f;  // Minimum YOLO confidence score
    public bool requireYoloValidation = true;       // Toggle YOLO validation on/off
    public bool debugYoloValidation = true;     
    [Header("ROI")]
    public bool roiFromCollidersFirst = true;
    [Range(0f, 0.3f)] public float roiPad01 = 0.08f;
    [Range(0.005f, 0.25f)] public float minRoiWH = 0.02f;
    public LayerMask sceneRaycastMask = ~0;
    
    [Header("Object Detection")]
    public string realWorldTag = "realWorld";        // Tag for real-world detected objects
    public string bimTag = "BIM";                    // Tag for BIM objects
    
    [Header("Settings")]
    public int minFeaturesForICP = 20;               // Minimum features needed to run ICP
    public bool runEveryFrame = true;
    public bool useROI = false;                      // Toggle ROI on/off
    public KeyCode manualTrigger = KeyCode.Space;
    [SerializeField] OpenAIBiasAdvisor biasAdvisor;
    [Header("Cooldown")]
    [Range(1f, 60f)] public float icpCooldownSeconds = 10f;  // Cooldown period per class in seconds
    public bool enableCooldown = true;               // Toggle to enable/disable cooldown
    
    [Header("Debug")]
    public bool debugLog = true;
    
    // Simple class matching dictionary
    Dictionary<string, Transform> realWorldObjects = new Dictionary<string, Transform>();
    Dictionary<string, Transform> bimObjects = new Dictionary<string, Transform>();
    
    // Cache for stable ROI calculation
    Dictionary<string, Rect> cachedROIs = new Dictionary<string, Rect>();
    Dictionary<string, Bounds> cachedBounds = new Dictionary<string, Bounds>();
    
    // Cooldown tracking
    Dictionary<string, float> lastAlignmentTime = new Dictionary<string, float>();
    
    // Cached feature provider component to avoid repeated AddComponent calls
    CustomFeatureProviderComponent featureProvider;
    static string NormalizeClass(string s) => LabelNorm.CanonicalKey(s);
    bool ValidateYoloDetection(string className, Transform realWorldObj, out Detection matchedDetection)
        {
            matchedDetection = null;
            className = NormalizeClass(className);
            if (!requireYoloValidation || yoloDetector == null)
            {
                if (debugYoloValidation)
                    Debug.Log($"YOLO validation disabled or detector missing for '{className}'");
                return false; // Skip validation if disabled
            }
            
            var detections = yoloDetector.LastDetections;
            if (detections == null || detections.Count == 0)
            {
                if (debugYoloValidation)
                    Debug.Log($"No YOLO detections available for '{className}'");
                return false;
            }
            
            // Get object's screen position for spatial validation
            Vector3 worldPos = realWorldObj.position;
            Vector3 screenPos = cam.WorldToScreenPoint(worldPos);
            
            // Convert to normalized coordinates (YOLO uses normalized coords)
            float normX = screenPos.x / Screen.width;
            float normY = 1f - (screenPos.y / Screen.height); // Flip Y for YOLO coordinate system
            
            // Find YOLO detections that match our class name
            Detection bestMatch = null;
            float bestScore = 0f;
            float bestDistance = float.MaxValue;
            
            foreach (var detection in detections)
            {
                // Check class name match (case insensitive)
                if (!LabelNorm.Match(className, detection.label)) continue;
                // Check confidence threshold
                if (detection.score < minYoloConfidence)
                {
                    if (debugYoloValidation)
                        Debug.Log($"YOLO detection '{detection.label}' below confidence threshold: {detection.score:F2} < {minYoloConfidence:F2}");
                    continue;
                }
                
                // Check spatial correspondence
                if (IsObjectInYoloBBox(normX, normY, detection, realWorldObj))
                {
                    // Calculate distance from object to bbox center for best match selection
                    float bboxCenterX = (detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) / 2f / yoloDetector.ImageWidth;
                    float bboxCenterY = (detection.bbox_xyxy[1] + detection.bbox_xyxy[3]) / 2f / yoloDetector.ImageHeight;
                    float distance = Vector2.Distance(new Vector2(normX, normY), new Vector2(bboxCenterX, bboxCenterY));
                    
                    if (detection.score > bestScore || (Mathf.Approximately(detection.score, bestScore) && distance < bestDistance))
                    {
                        bestMatch = detection;
                        bestScore = detection.score;
                        bestDistance = distance;
                    }
                }
            }
            
            if (bestMatch != null)
            {
                matchedDetection = bestMatch;
                if (debugYoloValidation)
                    Debug.Log($"✓ YOLO validated '{className}': confidence={bestMatch.score:F2}, label='{bestMatch.label}'");
                return true;
            }
            
            if (debugYoloValidation)
                Debug.Log($"✗ YOLO validation failed for '{className}': no matching detection with sufficient confidence");
            
            return false;
        }
        bool IsObjectInYoloBBox(float normX, float normY, Detection detection, Transform obj)
            {
                // Convert YOLO bbox to normalized coordinates
                float x1 = detection.bbox_xyxy[0] / (float)yoloDetector.ImageWidth;
                float y1 = detection.bbox_xyxy[1] / (float)yoloDetector.ImageHeight;
                float x2 = detection.bbox_xyxy[2] / (float)yoloDetector.ImageWidth;
                float y2 = detection.bbox_xyxy[3] / (float)yoloDetector.ImageHeight;
                
                // Add some tolerance for bbox matching
                float tolerance = 0.1f; // 10% tolerance
                x1 -= tolerance;
                y1 -= tolerance;
                x2 += tolerance;
                y2 += tolerance;
                
                bool isInside = normX >= x1 && normX <= x2 && normY >= y1 && normY <= y2;
                
                if (debugYoloValidation)
                {
                    Debug.Log($"Spatial check for {obj.name}: obj_pos=({normX:F3},{normY:F3}), bbox=({x1:F3},{y1:F3},{x2:F3},{y2:F3}) → {(isInside ? "INSIDE" : "OUTSIDE")}");
                }
                
                return isInside;
            }
    void Start()
    {
        if (!cornersDetector || !icp || !cam)
        {
            Debug.LogError("Missing required components!");
            enabled = false;
            return;
        }
        
        // Initialize feature provider component
        featureProvider = gameObject.GetComponent<CustomFeatureProviderComponent>();
        if (featureProvider == null)
            featureProvider = gameObject.AddComponent<CustomFeatureProviderComponent>();
        
        RefreshObjectLists();
        StartCoroutine(biasAdvisor.RefreshFromYaml(ok => { biasReady = ok; Debug.Log($"BiasAdvisor loaded={ok}"); }));
}
    
    void Update()
    {
        bool shouldRun = runEveryFrame || Input.GetKeyDown(manualTrigger);
        if (shouldRun)
        {
            RunAlignment();
        }
    }
    
    bool IsClassOnCooldown(string className)
    {
        if (!enableCooldown) return false;
        
        if (!lastAlignmentTime.TryGetValue(className, out float lastTime))
            return false; // Never aligned, not on cooldown
        
        float timeSinceLastAlignment = Time.time - lastTime;
        bool onCooldown = timeSinceLastAlignment < icpCooldownSeconds;
        
        if (debugLog && onCooldown)
        {
            float remaining = icpCooldownSeconds - timeSinceLastAlignment;
            Debug.Log($"Class '{className}' on cooldown - {remaining:F1}s remaining");
        }
        
        return onCooldown;
    }
    
    void UpdateAlignmentTime(string className)
    {
        lastAlignmentTime[className] = Time.time;
        
        if (debugLog)
            Debug.Log($"Updated alignment time for class '{className}' - next alignment available in {icpCooldownSeconds}s");
    }
    [ContextMenu("Debug YOLO Validation")]
void DebugYoloValidation()
{
    RefreshObjectLists();
    
    Debug.Log("=== YOLO VALIDATION DEBUG ===");
    
    if (yoloDetector == null)
    {
        Debug.LogError("No YOLO detector assigned!");
        return;
    }
    
    var detections = yoloDetector.LastDetections;
    Debug.Log($"Available YOLO detections: {detections?.Count ?? 0}");
    
    if (detections != null)
    {
        foreach (var detection in detections)
        {
            Debug.Log($"  YOLO: '{detection.label}' confidence={detection.score:F2} bbox=({detection.bbox_xyxy[0]},{detection.bbox_xyxy[1]},{detection.bbox_xyxy[2]},{detection.bbox_xyxy[3]})");
        }
    }
    
    Debug.Log($"Real-world objects to validate:");
    foreach (var kvp in realWorldObjects)
    {
        string className = kvp.Key;
        Transform obj = kvp.Value;
        
        bool isValid = ValidateYoloDetection(className, obj, out Detection match);
        string status = isValid ? "✅ VALID" : "❌ INVALID";
        
        Debug.Log($"  {status} '{className}' ({obj.name})");
        if (match != null)
        {
            Debug.Log($"    → Matched YOLO: '{match.label}' confidence={match.score:F2}");
        }
    }
}
    void RunAlignment()
    {
        RefreshObjectLists();
        
        int alignmentCount = 0;
        int cooldownSkippedCount = 0;
        int yoloValidationFailedCount = 0; // New counter
        
        // DEBUG: Show what we found
        if (debugLog)
        {
            Debug.Log("=== ALIGNMENT DEBUG (YOLO-Validated Child-to-Child Matching) ===");
            Debug.Log($"Real-world objects (including children): {realWorldObjects.Count}");
            Debug.Log($"BIM objects (including children): {bimObjects.Count}");
            Debug.Log($"YOLO validation: {(requireYoloValidation ? "ENABLED" : "DISABLED")}");
            if (requireYoloValidation && yoloDetector != null)
            {
                Debug.Log($"Available YOLO detections: {yoloDetector.LastDetections?.Count ?? 0}");
            }
        }

        // STEP 1: Do global feature detection (no ROI restrictions)
        var prevMask = cornersDetector.raycastMask;
        cornersDetector.raycastMask = sceneRaycastMask;
        cornersDetector.SetOnlyUnderRoot(null);
        cornersDetector.SetROI01(new Rect(0, 0, 1, 1));
        cornersDetector.ProcessOnceImmediate();
        
        var allWorldHits = cornersDetector.LastWorldHits;
        int totalFeatures = allWorldHits?.Count ?? 0;
        List<Vector3> allWorldHitsList = allWorldHits != null ? new List<Vector3>(allWorldHits) : new List<Vector3>();
        
        if (debugLog)
            Debug.Log($"Global feature detection: {totalFeatures} features found");
        
        // STEP 2: Pre-calculate stable ROIs for all real-world objects
        UpdateCachedBoundsAndROIs();
        
        // STEP 3: Process detected real-world children with YOLO validation
        foreach (var kvp in realWorldObjects)
        {
            string className = kvp.Key;
            Transform realWorldObj = kvp.Value;
            
            if (debugLog)
                Debug.Log($"\n--- Checking RW object: '{className}' ({realWorldObj.name}) ---");
            
            // COOLDOWN CHECK: Skip if class is on cooldown
            if (IsClassOnCooldown(className))
            {
                cooldownSkippedCount++;
                if (debugLog)
                    Debug.Log($"⏰ Class '{className}' on cooldown - skipping alignment");
                continue;
            }
            
            // YOLO VALIDATION: Check if YOLO actually detected this class
            if (!ValidateYoloDetection(className, realWorldObj, out Detection yoloMatch))
            {
                yoloValidationFailedCount++;
                if (debugLog)
                    Debug.Log($"🚫 YOLO validation failed for '{className}' - skipping ICP");
                continue;
            }
            
            // FEATURE CHECK: Verify sufficient features detected
            var objectFeatures = GetFeaturesForObject(allWorldHitsList, realWorldObj, className);
            
            if (debugLog)
                Debug.Log($"Features detected for this child: {objectFeatures.Count} (need {minFeaturesForICP})");
            
            if (objectFeatures.Count >= minFeaturesForICP)
            {
                if (debugLog)
                    Debug.Log($"✓ Child '{className}' has sufficient features - checking for BIM match...");
                
                // BIM MATCH CHECK: Find matching BIM child
                if (bimObjects.TryGetValue(className, out Transform bimObj))
                {
                    if (debugLog)
                    {
                        Debug.Log($"✓ Found matching BIM child: {bimObj.name} (Parent: {bimObj.parent?.name ?? "None"})");
                        Debug.Log($"✅ All checks passed - running ICP alignment...");
                        if (yoloMatch != null)
                            Debug.Log($"   YOLO: '{yoloMatch.label}' confidence={yoloMatch.score:F2}");
                    }
                    
                    // Ensure feature provider is available
                    if (featureProvider == null)
                    {
                        featureProvider = gameObject.GetComponent<CustomFeatureProviderComponent>();
                        if (featureProvider == null)
                            featureProvider = gameObject.AddComponent<CustomFeatureProviderComponent>();
                    }
                    
                    // Run ICP alignment
                    featureProvider.SetFeatures(objectFeatures);
                    icp.WorldHitsProvider = featureProvider;
                    icp.movementMode = ICPAligner.MovementMode.Standard;
                    icp.B = bimObj;
                float uprightB = biasAdvisor ? biasAdvisor.GetUprightBiasFor(className) : 0f;
                    if (use_bias)
                    {
                        uprightB = uprightB;
                    }
                    else {
                        uprightB = 0.0f;
                    }
                    icp.RunICP(uprightB*0.5f);
                    Debug.LogWarning(uprightB);
                    // var offset = new Vector3(-40.72f, 0f, 23.44f);
                    // bimObj.Translate(offset, Space.World); // Comment these out to see object overlap on ICP run
                    // Update cooldown
                    UpdateAlignmentTime(className);
                    alignmentCount++;
                    
                    if (debugLog)
                    {
                        Debug.Log($"✅ Successfully aligned '{className}': {objectFeatures.Count} features used");
                        Debug.Log($"   → Moved BIM child: {bimObj.name} to match YOLO-validated RW child: {realWorldObj.name}");
                    }
                }
                else
                {
                    if (debugLog)
                    {
                        Debug.Log($"✗ Child '{className}' validated by YOLO but no matching BIM child found");
                        Debug.Log("Available BIM classes: " + string.Join(", ", bimObjects.Keys));
                    }
                }
            }
            else
            {
                if (debugLog)
                    Debug.Log($"⚠ Child '{className}' YOLO-validated but insufficient features ({objectFeatures.Count}) - skipping ICP");
            }
        }
        
        // Reset detector
        cornersDetector.SetOnlyUnderRoot(null);
        cornersDetector.SetROI01(new Rect(0, 0, 1, 1));
        cornersDetector.raycastMask = prevMask;
        
        if (debugLog)
        {
            Debug.Log($"\n📊 ALIGNMENT SUMMARY:");
            Debug.Log($"✅ Successfully aligned: {alignmentCount}");
            Debug.Log($"⏰ Skipped (cooldown): {cooldownSkippedCount}");
            Debug.Log($"🚫 Failed YOLO validation: {yoloValidationFailedCount}");
            Debug.Log($"📝 Total children processed: {realWorldObjects.Count}");
            
            if (requireYoloValidation)
            {
                Debug.Log($"YOLO validation settings: min_confidence={minYoloConfidence:F2}, detections_available={yoloDetector?.LastDetections?.Count ?? 0}");
            }
        }
        
        // Reset detector
        cornersDetector.SetOnlyUnderRoot(null);
        cornersDetector.SetROI01(new Rect(0, 0, 1, 1));
        cornersDetector.raycastMask = prevMask;
        
        if (debugLog)
        {
            Debug.Log($"Alignment complete: {alignmentCount} DETECTED children aligned, {cooldownSkippedCount} skipped due to cooldown (out of {realWorldObjects.Count} total children)");
            if (enableCooldown && cooldownSkippedCount > 0)
                Debug.Log($"Cooldown active: {icpCooldownSeconds}s per class");
        }
    }
    
    void UpdateCachedBoundsAndROIs()
    {
        cachedBounds.Clear();
        cachedROIs.Clear();
        
        foreach (var kvp in realWorldObjects)
        {
            string className = kvp.Key;
            Transform obj = kvp.Value;
            
            // Calculate stable bounds
            Bounds bounds = GetBounds(obj);
            cachedBounds[className] = bounds;
            
            // Calculate stable ROI
            Rect roi = useROI ? GetViewportRect(bounds) : new Rect(0, 0, 1, 1);
            cachedROIs[className] = roi;
            
            if (debugLog && useROI)
                Debug.Log($"Cached ROI for '{className}': {roi}");
        }
    }
    
    List<Vector3> GetFeaturesForObject(List<Vector3> allFeatures, Transform realWorldObj, string className)
    {
        if (allFeatures == null || allFeatures.Count == 0)
            return new List<Vector3>();
        
        List<Vector3> objectFeatures = new List<Vector3>();
        
        if (useROI && cachedROIs.TryGetValue(className, out Rect roi))
        {
            // Filter features by ROI (viewport space)
            foreach (var worldHit in allFeatures)
            {
                Vector3 viewportPoint = cam.WorldToViewportPoint(worldHit);
                
                if (viewportPoint.z > 0 && // In front of camera
                    roi.Contains(new Vector2(viewportPoint.x, viewportPoint.y)))
                {
                    // Additional check: ensure feature is actually under this object's hierarchy
                    if (IsFeatureUnderObject(worldHit, realWorldObj))
                    {
                        objectFeatures.Add(worldHit);
                    }
                }
            }
        }
        else
        {
            // No ROI - filter only by object hierarchy
            foreach (var worldHit in allFeatures)
            {
                if (IsFeatureUnderObject(worldHit, realWorldObj))
                {
                    objectFeatures.Add(worldHit);
                }
            }
        }
        
        return objectFeatures;
    }
    
    bool IsFeatureUnderObject(Vector3 worldPoint, Transform rootObject)
    {
        // Use raycast or proximity check to see if feature belongs to this object
        // This is a simple proximity-based approach - you might want to use raycast instead
        
        if (cachedBounds.TryGetValue(GetClassNameFromTransform(rootObject), out Bounds bounds))
        {
            // Check if point is within bounds (with some tolerance)
            return bounds.Contains(worldPoint);
        }
        
        // Fallback: check distance to object
        float maxDistance = 2.0f; // Adjust based on your scene scale
        return Vector3.Distance(worldPoint, rootObject.position) <= maxDistance;
    }
    
    string GetClassNameFromTransform(Transform t)
    {
        // Helper to get class name from transform (reverse lookup)
        foreach (var kvp in realWorldObjects)
        {
            if (kvp.Value == t)
                return kvp.Key;
        }
        return GetClassName(t.gameObject);
    }
    
    void RefreshObjectLists()
    {
        realWorldObjects.Clear();
        bimObjects.Clear();
        
        // Find all parent objects with realWorld tag, then process their children
        GameObject[] realWorldParents = GameObject.FindGameObjectsWithTag(realWorldTag);
        foreach (var parentObj in realWorldParents)
        {
            // Process parent itself
            string parentClassName = GetClassName(parentObj);
            if (!string.IsNullOrEmpty(parentClassName))
            {
                realWorldObjects[parentClassName] = parentObj.transform;
                if (debugLog)
                    Debug.Log($"RW Parent: '{parentObj.name}' → Class: '{parentClassName}'");
            }
            
            // Process all children recursively
            ProcessChildrenRecursively(parentObj.transform, realWorldObjects, "RW");
        }
        
        // Find all parent objects with BIM tag, then process their children
        GameObject[] bimParents = GameObject.FindGameObjectsWithTag(bimTag);
        foreach (var parentObj in bimParents)
        {
            // Process parent itself
            string parentClassName = GetClassName(parentObj);
            if (!string.IsNullOrEmpty(parentClassName))
            {
                bimObjects[parentClassName] = parentObj.transform;
                if (debugLog)
                    Debug.Log($"BIM Parent: '{parentObj.name}' → Class: '{parentClassName}'");
            }
            
            // Process all children recursively
            ProcessChildrenRecursively(parentObj.transform, bimObjects, "BIM");
        }
        
        if (debugLog)
            Debug.Log($"Found {realWorldObjects.Count} real-world objects (including children), {bimObjects.Count} BIM objects (including children)");
    }
    
    void ProcessChildrenRecursively(Transform parent, Dictionary<string, Transform> targetDict, string type)
    {
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            string childClassName = GetClassName(child.gameObject);
            
            if (!string.IsNullOrEmpty(childClassName))
            {
                // Use unique key to avoid conflicts (parent_child or just child if unique)
                string uniqueKey = childClassName;
                
                // If key already exists, make it unique by adding parent info
                if (targetDict.ContainsKey(uniqueKey))
                {
                    string parentClass = GetClassName(parent.gameObject);
                    uniqueKey = $"{parentClass}_{childClassName}";
                    
                    // If still conflicts, add index
                    int counter = 1;
                    string baseKey = uniqueKey;
                    while (targetDict.ContainsKey(uniqueKey))
                    {
                        uniqueKey = $"{baseKey}_{counter}";
                        counter++;
                    }
                }
                
                targetDict[uniqueKey] = child;
                
                if (debugLog)
                    Debug.Log($"  {type} Child: '{child.name}' → Class: '{uniqueKey}'");
            }
            
            // Recursively process grandchildren
            ProcessChildrenRecursively(child, targetDict, type);
        }
    }
    
string GetClassName(GameObject obj)
{
    string name = obj.name.ToLower();
    
    // Remove prefixes/suffixes
    string[] prefixes = { "rw_", "bim_", "detected_", "virtual_" };
    foreach (string prefix in prefixes)
    {
        if (name.StartsWith(prefix))
        {
            name = name.Substring(prefix.Length);
            break;
        }
    }
    
    string[] suffixes = { "_world", "_bim", "_real", "_virtual", "_detected" };
    foreach (string suffix in suffixes)
    {
        if (name.EndsWith(suffix))
        {
            name = name.Substring(0, name.Length - suffix.Length);
            break;
        }
    }
    
    // Clean up any instance numbers or special characters
    name = System.Text.RegularExpressions.Regex.Replace(name, @"[^a-z]", "");
    
   return NormalizeClass(name);
}
    
    bool TryCombinedBounds(Transform obj, bool collidersFirst, LayerMask mask, out Bounds bounds)
    {
        bounds = new Bounds();
        
        if (obj == null) return false;
        
        if (collidersFirst)
        {
            // Try colliders first
            Collider[] colliders = obj.GetComponentsInChildren<Collider>();
            if (colliders.Length > 0)
            {
                bounds = colliders[0].bounds;
                for (int i = 1; i < colliders.Length; i++)
                {
                    bounds.Encapsulate(colliders[i].bounds);
                }
                return true;
            }
        }
        
        // Try renderers
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        if (renderers.Length > 0)
        {
            bounds = renderers[0].bounds;
            for (int i = 1; i < renderers.Length; i++)
            {
                bounds.Encapsulate(renderers[i].bounds);
            }
            return true;
        }
        
        return false;
    }
    
    bool TryViewportRectFromBounds(Bounds bounds, Camera camera, float padding, float minSize, out Rect rect)
    {
        rect = GetViewportRect(bounds);
        
        // Apply padding
        rect.x -= padding;
        rect.y -= padding;
        rect.width += padding * 2;
        rect.height += padding * 2;
        
        // Ensure minimum size
        if (rect.width < minSize)
        {
            float diff = minSize - rect.width;
            rect.x -= diff * 0.5f;
            rect.width = minSize;
        }
        if (rect.height < minSize)
        {
            float diff = minSize - rect.height;
            rect.y -= diff * 0.5f;
            rect.height = minSize;
        }
        
        // Clamp to viewport
        rect.x = Mathf.Max(0, rect.x);
        rect.y = Mathf.Max(0, rect.y);
        rect.width = Mathf.Min(1 - rect.x, rect.width);
        rect.height = Mathf.Min(1 - rect.y, rect.height);
        
        return rect.width > 0 && rect.height > 0;
    }
    
    Bounds GetBounds(Transform obj)
    {
        // Try to get bounds from renderers first (including children)
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        if (renderers.Length > 0)
        {
            Bounds bounds = renderers[0].bounds;
            for (int i = 1; i < renderers.Length; i++)
            {
                bounds.Encapsulate(renderers[i].bounds);
            }
            
            if (debugLog)
                Debug.Log($"Bounds from {renderers.Length} renderers: {bounds.center}, size: {bounds.size}");
            
            return bounds;
        }
        
        // Fallback to colliders (including children)
        Collider[] colliders = obj.GetComponentsInChildren<Collider>();
        if (colliders.Length > 0)
        {
            Bounds bounds = colliders[0].bounds;
            for (int i = 1; i < colliders.Length; i++)
            {
                bounds.Encapsulate(colliders[i].bounds);
            }
            
            if (debugLog)
                Debug.Log($"Bounds from {colliders.Length} colliders: {bounds.center}, size: {bounds.size}");
            
            return bounds;
        }
        
        // If no renderers or colliders, try to calculate from child positions
        if (obj.childCount > 0)
        {
            Vector3 min = obj.GetChild(0).position;
            Vector3 max = obj.GetChild(0).position;
            
            for (int i = 0; i < obj.childCount; i++)
            {
                Vector3 pos = obj.GetChild(i).position;
                min = Vector3.Min(min, pos);
                max = Vector3.Max(max, pos);
            }
            
            Vector3 center = (min + max) * 0.5f;
            Vector3 size = max - min;
            
            // Add some padding if size is too small
            if (size.magnitude < 0.1f)
                size = Vector3.one;
            
            if (debugLog)
                Debug.Log($"Bounds from child positions: {center}, size: {size}");
            
            return new Bounds(center, size);
        }
        
        // Last resort: just use position with default size
        if (debugLog)
            Debug.Log($"Using fallback bounds at {obj.position}");
        
        return new Bounds(obj.position, Vector3.one);
    }
    
    Rect GetViewportRect(Bounds bounds)
    {
        // Convert 3D bounds to 2D viewport rectangle
        Vector3[] corners = new Vector3[8];
        Vector3 center = bounds.center;
        Vector3 extents = bounds.extents;
        
        // Get all 8 corners of the bounding box
        int idx = 0;
        for (int x = -1; x <= 1; x += 2)
        for (int y = -1; y <= 1; y += 2)
        for (int z = -1; z <= 1; z += 2)
        {
            corners[idx++] = center + Vector3.Scale(extents, new Vector3(x, y, z));
        }
        
        // Project to viewport and find min/max
        float minX = 1f, minY = 1f, maxX = 0f, maxY = 0f;
        bool anyVisible = false;
        
        foreach (Vector3 corner in corners)
        {
            Vector3 vp = cam.WorldToViewportPoint(corner);
            if (vp.z > 0) // In front of camera
            {
                anyVisible = true;
                minX = Mathf.Min(minX, vp.x);
                minY = Mathf.Min(minY, vp.y);
                maxX = Mathf.Max(maxX, vp.x);
                maxY = Mathf.Max(maxY, vp.y);
            }
        }
        
        if (!anyVisible)
            return new Rect(0, 0, 1, 1); // Full viewport if nothing visible
        
        // Clamp to viewport bounds and add small padding
        float padding = 0.05f;
        minX = Mathf.Max(0, minX - padding);
        minY = Mathf.Max(0, minY - padding);
        maxX = Mathf.Min(1, maxX + padding);
        maxY = Mathf.Max(1, maxY + padding);
        
        return new Rect(minX, minY, maxX - minX, maxY - minY);
    }
    
    // MonoBehaviour feature provider for filtered features (required since ICPAligner expects UnityEngine.Object)
    public class CustomFeatureProviderComponent : MonoBehaviour
    {
        private List<Vector3> features = new List<Vector3>();
        
        public void SetFeatures(List<Vector3> featureList)
        {
            features = featureList ?? new List<Vector3>();
        }
        
        // Property that matches what your ICPAligner expects
        public IReadOnlyList<Vector3> LastWorldHits => features;
        
        // Alternative methods in case your ICP system needs different interfaces
        public List<Vector3> GetFeatures() => features;
        public int Count => features.Count;
        public Vector3 this[int index] => features[index];
        
        // In case your ICP system expects IList interface
        public IList<Vector3> GetAsList() => features;
    }
    
    [ContextMenu("Manual Run")]
    void ManualRun()
    {
        RunAlignment();
    }
    
    [ContextMenu("Clear All Cooldowns")]
    void ClearAllCooldowns()
    {
        lastAlignmentTime.Clear();
        Debug.Log("All class cooldowns cleared - all classes can now be aligned immediately");
    }
    
    [ContextMenu("Show Cooldown Status")]
    void ShowCooldownStatus()
    {
        if (!enableCooldown)
        {
            Debug.Log("Cooldown system is DISABLED");
            return;
        }
        
        Debug.Log($"=== COOLDOWN STATUS (Duration: {icpCooldownSeconds}s) ===");
        
        if (lastAlignmentTime.Count == 0)
        {
            Debug.Log("No classes have been aligned yet - all classes available");
            return;
        }
        
        float currentTime = Time.time;
        int onCooldownCount = 0;
        int availableCount = 0;
        
        foreach (var kvp in lastAlignmentTime)
        {
            string className = kvp.Key;
            float lastTime = kvp.Value;
            float timeSinceLastAlignment = currentTime - lastTime;
            bool onCooldown = timeSinceLastAlignment < icpCooldownSeconds;
            
            if (onCooldown)
            {
                float remaining = icpCooldownSeconds - timeSinceLastAlignment;
                Debug.Log($"⏰ '{className}' - {remaining:F1}s remaining");
                onCooldownCount++;
            }
            else
            {
                Debug.Log($"✅ '{className}' - available");
                availableCount++;
            }
        }
        
        Debug.Log($"Summary: {onCooldownCount} on cooldown, {availableCount} available");
    }
    
    [ContextMenu("List Detected Classes")]
    void ListClasses()
    {
        RefreshObjectLists();
        
        Debug.Log("=== Real World Objects (Parents + Children) ===");
        foreach (var kvp in realWorldObjects)
        {
            Debug.Log($"  Class: '{kvp.Key}' -> {kvp.Value.name} (Parent: {kvp.Value.parent?.name ?? "Root"})");
        }
        
        Debug.Log("=== BIM Objects (Parents + Children) ===");
        foreach (var kvp in bimObjects)
        {
            Debug.Log($"  Class: '{kvp.Key}' -> {kvp.Value.name} (Parent: {kvp.Value.parent?.name ?? "Root"})");
        }
        
        Debug.Log("=== Child-to-Child Matching Pairs ===");
        int matchCount = 0;
        foreach (var kvp in realWorldObjects)
        {
            if (bimObjects.ContainsKey(kvp.Key))
            {
                var rwChild = kvp.Value;
                var bimChild = bimObjects[kvp.Key];
                Debug.Log($"  ✓ Match: '{kvp.Key}'");
                Debug.Log($"    RW Child:  {rwChild.name} (Parent: {rwChild.parent?.name ?? "Root"})");
                Debug.Log($"    BIM Child: {bimChild.name} (Parent: {bimChild.parent?.name ?? "Root"})");
                matchCount++;
            }
        }
        Debug.Log($"Total child-to-child matches: {matchCount}");
    }
    
    [ContextMenu("Debug All Objects")]
    void DebugAllObjects()
    {
        Debug.Log("=== ALL GAMEOBJECTS IN SCENE ===");
        GameObject[] allObjects = FindObjectsByType<GameObject>(FindObjectsSortMode.None);
        
        foreach (GameObject obj in allObjects)
        {
            string className = GetClassName(obj);
            Debug.Log($"Object: '{obj.name}' | Tag: '{obj.tag}' | Class: '{className}' | Active: {obj.activeInHierarchy}");
        }
        
        Debug.Log($"\nTotal objects found: {allObjects.Length}");
    }

    [ContextMenu("Debug Tagged Objects")]
    void DebugTaggedObjects()
    {
        Debug.Log("=== CHECKING TAGGED OBJECTS ===");
        
        // Check realWorld tagged objects
        GameObject[] realWorldObjs = GameObject.FindGameObjectsWithTag(realWorldTag);
        Debug.Log($"Objects with '{realWorldTag}' tag: {realWorldObjs.Length}");
        foreach (var obj in realWorldObjs)
        {
            Debug.Log($"  - {obj.name} (Active: {obj.activeInHierarchy})");
        }
        
        // Check BIM tagged objects
        GameObject[] bimObjs = GameObject.FindGameObjectsWithTag(bimTag);
        Debug.Log($"Objects with '{bimTag}' tag: {bimObjs.Length}");
        foreach (var obj in bimObjs)
        {
            Debug.Log($"  - {obj.name} (Active: {obj.activeInHierarchy})");
        }
    }

    [ContextMenu("Test Class Name Matching")]
    void TestClassNameMatching()
    {
        RefreshObjectLists();
        
        Debug.Log("=== CHILD-TO-CHILD CLASS NAME MATCHING TEST ===");
        
        Debug.Log("Real-world object class names (parents + children):");
        foreach (var kvp in realWorldObjects)
        {
            var obj = kvp.Value;
            Debug.Log($"  '{obj.name}' → Class: '{kvp.Key}' (Parent: {obj.parent?.name ?? "Root"})");
        }
        
        Debug.Log("BIM object class names (parents + children):");
        foreach (var kvp in bimObjects)
        {
            var obj = kvp.Value;
            Debug.Log($"  '{obj.name}' → Class: '{kvp.Key}' (Parent: {obj.parent?.name ?? "Root"})");
        }
        
        Debug.Log("Child-to-child matches found:");
        int matchCount = 0;
        foreach (var rwKvp in realWorldObjects)
        {
            if (bimObjects.ContainsKey(rwKvp.Key))
            {
                var rwChild = rwKvp.Value;
                var bimChild = bimObjects[rwKvp.Key];
                Debug.Log($"  ✓ Match: '{rwKvp.Key}'");
                Debug.Log($"    RW:  {rwChild.name} (in {rwChild.parent?.name ?? "Root"})");
                Debug.Log($"    BIM: {bimChild.name} (in {bimChild.parent?.name ?? "Root"})");
                matchCount++;
            }
            else
            {
                Debug.Log($"  ✗ No match for RW class: '{rwKvp.Key}' ({rwKvp.Value.name})");
            }
        }
        
        Debug.Log($"Total child-to-child matches: {matchCount}");
    }
    
    [ContextMenu("Test Feature Detection")]
    void TestFeatureDetection()
    {
        RefreshObjectLists();
        UpdateCachedBoundsAndROIs();
        
        Debug.Log("=== FEATURE DETECTION TEST ===");
        
        // Test global detection
        Debug.Log("Testing global feature detection...");
        var prevMask = cornersDetector.raycastMask;
        cornersDetector.raycastMask = sceneRaycastMask;
        cornersDetector.SetROI01(new Rect(0, 0, 1, 1));
        cornersDetector.SetOnlyUnderRoot(null);
        cornersDetector.ProcessOnceImmediate();
        var allFeatures = cornersDetector.LastWorldHits;
        int totalFeatures = allFeatures?.Count ?? 0;
        Debug.Log($"Global features detected: {totalFeatures}");
        
        // Convert to List for easier manipulation
        List<Vector3> allFeaturesList = allFeatures != null ? new List<Vector3>(allFeatures) : new List<Vector3>();
        
        // Test filtering for each object
        foreach (var kvp in realWorldObjects)
        {
            string className = kvp.Key;
            Transform rwObj = kvp.Value;
            
            var objectFeatures = GetFeaturesForObject(allFeaturesList, rwObj, className);
            bool isDetected = objectFeatures.Count >= minFeaturesForICP;
            bool isOnCooldown = IsClassOnCooldown(className);
            
            string status = isOnCooldown ? "⏰ ON COOLDOWN" : 
                           isDetected ? "✓ DETECTED" : 
                           "✗ NOT DETECTED";
            
            Debug.Log($"Child '{className}' ({rwObj.name}): {objectFeatures.Count} features - {status}");
            
            if (cachedROIs.TryGetValue(className, out Rect roi))
            {
                Debug.Log($"  ROI: {roi}");
            }
        }
        
        // Reset
        cornersDetector.SetROI01(new Rect(0, 0, 1, 1));
        cornersDetector.SetOnlyUnderRoot(null);
        cornersDetector.raycastMask = prevMask;
    }
    
    [ContextMenu("Show Detection Status")]
    void ShowDetectionStatus()
    {
        RefreshObjectLists();
        
        Debug.Log("=== DETECTION STATUS ===");
        
        // Do a quick global detection
        var prevMask = cornersDetector.raycastMask;
        cornersDetector.raycastMask = sceneRaycastMask;
        cornersDetector.SetROI01(new Rect(0, 0, 1, 1));
        cornersDetector.SetOnlyUnderRoot(null);
        cornersDetector.ProcessOnceImmediate();
        var allFeatures = cornersDetector.LastWorldHits;
        List<Vector3> allFeaturesList = allFeatures != null ? new List<Vector3>(allFeatures) : new List<Vector3>();
        
        int detectedCount = 0;
        int matchedCount = 0;
        int alignableCount = 0;
        int onCooldownCount = 0;
        
        foreach (var kvp in realWorldObjects)
        {
            string className = kvp.Key;
            Transform rwObj = kvp.Value;
            
            var objectFeatures = GetFeaturesForObject(allFeaturesList, rwObj, className);
            bool isDetected = objectFeatures.Count >= minFeaturesForICP;
            bool hasMatch = bimObjects.ContainsKey(className);
            bool isOnCooldown = IsClassOnCooldown(className);
            bool isAlignable = isDetected && hasMatch && !isOnCooldown;
            
            if (isDetected) detectedCount++;
            if (hasMatch) matchedCount++;
            if (isAlignable) alignableCount++;
            if (isOnCooldown) onCooldownCount++;
            
            string status = isOnCooldown ? "⏰ ON COOLDOWN" :
                           isAlignable ? "🎯 WILL ALIGN" : 
                           isDetected ? "👁 DETECTED" : 
                           hasMatch ? "🔗 MATCHED" : "⚪ IDLE";
            
            Debug.Log($"{status} '{className}' ({rwObj.name}) - Features: {objectFeatures.Count}, BIM Match: {hasMatch}");
        }
        
        Debug.Log($"\n📊 SUMMARY:");
        Debug.Log($"Total children: {realWorldObjects.Count}");
        Debug.Log($"Detected: {detectedCount}");
        Debug.Log($"Have BIM matches: {matchedCount}");
        Debug.Log($"On cooldown: {onCooldownCount}");
        Debug.Log($"Will align: {alignableCount}");
        Debug.Log($"Cooldown enabled: {enableCooldown} ({icpCooldownSeconds}s)");
        
        // Reset
        cornersDetector.SetROI01(new Rect(0, 0, 1, 1));
        cornersDetector.SetOnlyUnderRoot(null);
        cornersDetector.raycastMask = prevMask;
    }
}