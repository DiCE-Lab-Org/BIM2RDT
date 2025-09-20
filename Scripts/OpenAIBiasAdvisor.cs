using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public interface ILLMAdvisor { string Complete(string systemPrompt, string userPrompt); }

[DisallowMultipleComponent]
public class OpenAIBiasAdvisor : MonoBehaviour, ILLMAdvisor
{
    static string Key(string s) => SimpleBoundingBoxICP.LabelNorm.CanonicalKey(s);

    [Header("OpenAI")]
    [SerializeField] string apiKey;                             // put your key here
    [SerializeField] string model = "gpt-4o-mini-2024-07-18";   // supports response_format=json_object
    [Range(0, 1)] public float defaultUprightBias = 0.3f;       // was 2.0f vs [Range(0,1)] -> invalid

    [Header("YAML")]
    [Tooltip("YAML TextAsset with classes, e.g.\ntext:\n- cone\n- woodencrate\n- barrel")]
    public TextAsset yamlFile;

    [Serializable] public class BiasItem { public string label; public float upright_bias; public bool yaw_only; }
    [Serializable] public class BiasItems { public List<BiasItem> items; }

    [ContextMenu("Dump Bias Cache")]
    void DumpBiasCache()
    {
        foreach (var kv in cache)
            Debug.Log($"{kv.Key}  bias={kv.Value.upright_bias:F2}  yaw_only={kv.Value.yaw_only}");
    }

    readonly Dictionary<string, BiasItem> cache = new(StringComparer.Ordinal);

    // -------- Public API --------
    public IEnumerator RefreshFromYaml(Action<bool> done = null)
    {
        cache.Clear();

        var labels = ParseYamlLabels(yamlFile ? yamlFile.text : "");
        if (labels.Count == 0) { done?.Invoke(false); yield break; }

        string system = "You return BIM alignment priors. For each label, output: "
                      + "upright_bias ∈ [0,1] (fraction of tilt corrected per ICP iteration) "
                      + "and yaw_only (true if object should be treated as yaw-only due to vertical symmetry or gravity alignment). "
                      + "Only include labels given. No extra text.";

        string user = "Labels (YAML below)."
                    + "\nRules:"
                    + "\n- Furniture, crates, cabinets: higher upright_bias (≈0.3–0.7)."
                    + "\n- Cylinders/cones/barrels: yaw_only=true if standing; modest bias (≈0.1–0.4)."
                    + "\n- Spheres or fully isotropic: yaw_only=false, very low bias (≈0.0–0.1)."
                    + "\n- Take gravity into account, how likely is this to not be upright? if it is likely we need a smaller value."
                    + "\nReturn JSON that matches the provided schema strictly."
                    + "\n\nYAML:\n" + (yamlFile ? yamlFile.text : "text: []");

        // Request DTO
        var req = new ChatRequest
        {
            model = model,
            temperature = 0,
            response_format = new ResponseFormat { type = "json_object" }, // let the model emit pure JSON
            messages = new[]
            {
                new Msg{ role="system", content=system },
                new Msg{ role="user",   content=BuildUserPrompt(labels, user) }
            }
        };

        string url = "https://api.openai.com/v1/chat/completions";
        string body = JsonUtility.ToJson(req, false);

        using (var www = new UnityWebRequest(url, "POST"))
        {
            byte[] payload = Encoding.UTF8.GetBytes(body);
            www.uploadHandler   = new UploadHandlerRaw(payload);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Authorization", "Bearer " + apiKey);
            www.SetRequestHeader("Content-Type", "application/json");

            var op = www.SendWebRequest();
            while (!op.isDone) yield return null;

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"OpenAI error: {www.responseCode} {www.error}\n{www.downloadHandler.text}");
                done?.Invoke(false); yield break;
            }

            var resp = JsonUtility.FromJson<ChatResponse>(www.downloadHandler.text);
            string content = resp != null && resp.choices != null && resp.choices.Length > 0
                ? resp.choices[0]?.message?.content
                : null;

            if (string.IsNullOrEmpty(content))
                content = SliceJson(www.downloadHandler.text); // fallback slice

            BiasItems parsed = null;
            try { parsed = JsonUtility.FromJson<BiasItems>(content); }
            catch { parsed = JsonUtility.FromJson<BiasItems>(SliceJson(content)); }

            if (parsed == null || parsed.items == null)
            {
                Debug.LogError("Failed to parse JSON from model:\n" + content);
                done?.Invoke(false); yield break;
            }

            foreach (var it in parsed.items)
            {
                if (it == null || string.IsNullOrWhiteSpace(it.label)) continue;
                it.upright_bias = Mathf.Clamp01(it.upright_bias);
                cache[Key(it.label)] = it;
            }

            done?.Invoke(true);
        }
    }

    public float GetUprightBiasFor(string label)
    {
        if (cache.TryGetValue(Key(label), out var it)) return Mathf.Clamp01(it.upright_bias);
        return Mathf.Clamp01(defaultUprightBias);
    }

    public bool GetYawOnlyFor(string label)
    {
        return cache.TryGetValue(Key(label), out var it) && it.yaw_only;
    }

    // ILLMAdvisor simple passthrough
    public string Complete(string systemPrompt, string userPrompt) => "{\"ok\":true}";

    // -------- Helpers --------
    static List<string> ParseYamlLabels(string yaml)
    {
        var outList = new List<string>();
        if (string.IsNullOrWhiteSpace(yaml)) return outList;
        bool inList = false;
        foreach (var raw in yaml.Split('\n'))
        {
            var s = raw.Trim();
            if (s.StartsWith("text:")) { inList = true; continue; }
            if (!inList) continue;
            if (s.StartsWith("-")) outList.Add(s.Substring(1).Trim());
            else if (s.Length == 0) break;
        }
        return outList;
    }

    static string BuildUserPrompt(List<string> labels, string preface)
    {
        var sb = new StringBuilder(preface.Length + labels.Count * 16);
        sb.Append(preface).Append("\n\nLabels:\n");
        foreach (var l in labels) sb.Append("- ").Append(l).Append('\n');
        sb.Append("\nReturn: { \"items\": [ {\"label\":\"...\",\"upright_bias\":0.0,\"yaw_only\":false}, ... ] }");
        return sb.ToString();
    }

    static string SliceJson(string s)
    {
        int a = s.IndexOf('{'); int b = s.LastIndexOf('}');
        return (a >= 0 && b > a) ? s.Substring(a, b - a + 1) : "{}";
    }

    // ----- DTOs for Chat Completions -----
    [Serializable] class Msg { public string role; public string content; }

    [Serializable] class ChatRequest
    {
        public string model;
        public float temperature;
        public Msg[] messages;
        public ResponseFormat response_format;
    }

    [Serializable] class ResponseFormat { public string type; }

    [Serializable] class ChatResponse { public Choice[] choices; }
    [Serializable] class Choice { public Message message; }
    [Serializable] class Message { public string content; }
}
