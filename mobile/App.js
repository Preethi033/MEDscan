import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";

const API_URL = (process.env.EXPO_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);
const NOTICE =
  "Research prototype only — not for diagnosis, treatment decisions, or emergencies.";
const TABS = [
  ["review", "Review"],
  ["samples", "Samples"],
  ["failures", "Failures"],
  ["about", "Limitations"],
];

export default function App() {
  const [tab, setTab] = useState("review");
  const [serverStatus, setServerStatus] = useState("Checking research API…");
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [examples, setExamples] = useState([]);
  const [failures, setFailures] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then(readJson)
      .then((health) =>
        setServerStatus(
          `API connected · ${Object.values(health.features || {}).filter(Boolean).length} model components configured`,
        ),
      )
      .catch(() => setServerStatus("API unavailable · start the Python server on this network"));
  }, []);

  async function chooseImage() {
    try {
      setError("");
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        setError("Photo-library permission is needed to select a sample image.");
        return;
      }
      const selection = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["images"],
        quality: 1,
        allowsEditing: false,
      });
      if (selection.canceled || !selection.assets?.length) return;
      setImage(selection.assets[0]);
      setResult(null);
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function analyzeImage() {
    if (!image) {
      setError("Choose an image before requesting analysis.");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      const filename = image.fileName || "research-sample.jpg";
      if (Platform.OS === "web") {
        const response = await fetch(image.uri);
        if (!response.ok) {
          throw new Error("Could not read the selected image from the browser.");
        }
        body.append("file", await response.blob(), filename);
      } else {
        body.append("file", {
          uri: image.uri,
          name: filename,
          type: image.mimeType || "image/jpeg",
        });
      }
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        body,
      });
      setResult(await readJson(response));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadExamples() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/samples?limit=8`);
      const payload = await readJson(response);
      setExamples(payload.examples || []);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadFailures() {
    setBusy(true);
    setError("");
    try {
      setFailures(await readJson(await fetch(`${API_URL}/api/failures`)));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  function changeTab(nextTab) {
    setTab(nextTab);
    setError("");
    if (nextTab === "samples" && examples.length === 0) loadExamples();
    if (nextTab === "failures" && failures === null) loadFailures();
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <View style={styles.header}>
        <Text style={styles.eyebrow}>MEDICAL AI · RESEARCH DEMO</Text>
        <Text style={styles.title}>ScanExplain</Text>
        <Text style={styles.subtitle}>Explore model behavior, not clinical answers.</Text>
      </View>

      <View style={styles.notice}>
        <Text style={styles.noticeText}>{NOTICE}</Text>
      </View>
      <Text style={styles.connection}>{serverStatus}</Text>

      <View style={styles.tabRow}>
        {TABS.map(([id, label]) => (
          <Pressable
            accessibilityRole="tab"
            accessibilityState={{ selected: tab === id }}
            key={id}
            onPress={() => changeTab(id)}
            style={[styles.tab, tab === id && styles.activeTab]}
          >
            <Text style={[styles.tabText, tab === id && styles.activeTabText]}>{label}</Text>
          </Pressable>
        ))}
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        {error ? <Text style={styles.error}>{error}</Text> : null}
        {busy ? <ActivityIndicator color="#087e8b" style={styles.spinner} /> : null}

        {tab === "review" ? (
          <ReviewPanel
            image={image}
            result={result}
            onChoose={chooseImage}
            onAnalyze={analyzeImage}
          />
        ) : null}
        {tab === "samples" ? (
          <SamplesPanel examples={examples} onReload={loadExamples} />
        ) : null}
        {tab === "failures" ? <FailuresPanel report={failures} /> : null}
        {tab === "about" ? <LimitationsPanel /> : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function ReviewPanel({ image, result, onChoose, onAnalyze }) {
  return (
    <View>
      <Text style={styles.sectionTitle}>Review a sample image</Text>
      <Text style={styles.body}>
        Select a supported research sample. The current backend is stub-first and will say
        plainly when a model component is not connected. Uploads must be JPEG, PNG, or WebP,
        no larger than 12 MB or 4096 pixels per side.
      </Text>
      <Pressable onPress={onChoose} style={styles.secondaryButton}>
        <Text style={styles.secondaryButtonText}>Choose image</Text>
      </Pressable>
      {image ? (
        <Image source={{ uri: image.uri }} style={styles.preview} resizeMode="contain" />
      ) : (
        <View style={styles.emptyImage}>
          <Text style={styles.muted}>No image selected</Text>
        </View>
      )}
      <Pressable
        accessibilityRole="button"
        onPress={onAnalyze}
        style={[styles.primaryButton, !image && styles.disabledButton]}
      >
        <Text style={styles.primaryButtonText}>Analyze research sample</Text>
      </Pressable>
      {result ? <AnalysisResult result={result} /> : null}
    </View>
  );
}

function AnalysisResult({ result }) {
  const prediction = result.prediction;
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>
        {prediction ? `Model label: ${prediction.label}` : "No model prediction"}
      </Text>
      {prediction ? (
        <>
          <Text style={styles.body}>
            Top-class confidence: {(result.confidence * 100).toFixed(1)}%
          </Text>
          <Text style={styles.body}>
            Class probabilities: {JSON.stringify(prediction.probs)}
          </Text>
        </>
      ) : (
        <Text style={styles.body}>The classifier is not connected; no prediction was made.</Text>
      )}
      <Text style={styles.cardTitle}>Upload screening</Text>
      <Text style={styles.body}>
        {result.screening?.status || "Unavailable until a prediction is produced"}
      </Text>
      {result.screening?.reasons?.length ? (
        <Text style={styles.body}>Notes: {result.screening.reasons.join(", ")}</Text>
      ) : null}
      <OverlayResult title="Explanation overlay" uri={result.explanation_overlay} />
      <OverlayResult title="Segmentation overlay" uri={result.segmentation_overlay} />
      {result.comparison ? (
        <View>
          <Text style={styles.cardTitle}>Labeled training example</Text>
          <Text style={styles.body}>{result.comparison.label}</Text>
          <Image source={{ uri: result.comparison.image }} style={styles.overlay} resizeMode="contain" />
        </View>
      ) : (
        <Text style={styles.muted}>Labeled-example comparison is not connected.</Text>
      )}
    </View>
  );
}

function OverlayResult({ title, uri }) {
  return (
    <View>
      <Text style={styles.cardTitle}>{title}</Text>
      {uri ? (
        <Image source={{ uri }} style={styles.overlay} resizeMode="contain" />
      ) : (
        <Text style={styles.muted}>Not available — no component is connected.</Text>
      )}
    </View>
  );
}

function SamplesPanel({ examples, onReload }) {
  return (
    <View>
      <Text style={styles.sectionTitle}>PneumoniaMNIST benchmark</Text>
      <Text style={styles.body}>
        224×224 downsampled pediatric chest X-ray classification samples. Training split only;
        no segmentation masks. Not intended for clinical use. First load downloads the official
        CC BY 4.0 dataset to the Python server cache.
      </Text>
      <Pressable onPress={onReload} style={styles.secondaryButton}>
        <Text style={styles.secondaryButtonText}>Load sample images</Text>
      </Pressable>
      {examples.map((example, index) => (
        <View key={`${index}-${example.label}`} style={styles.sampleCard}>
          <Image source={{ uri: example.image }} style={styles.sampleImage} resizeMode="contain" />
          <Text style={styles.body}>{example.label}</Text>
        </View>
      ))}
      <Text style={styles.muted}>
        Samples show dataset labels, not model outputs. See medmnist.com and Zenodo record
        10519652 for attribution.
      </Text>
    </View>
  );
}

function FailuresPanel({ report }) {
  return (
    <View>
      <Text style={styles.sectionTitle}>Failure gallery</Text>
      <Text style={styles.body}>
        Review failures from labeled test-set evaluation, including low-confidence and
        human-annotated shortcut-cue cases.
      </Text>
      {!report || report.status === "not_connected" ? (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>No evaluation report connected</Text>
          <Text style={styles.body}>
            This is not evidence that the model has no errors. Connect the team's labeled
            test-set report to populate this view.
          </Text>
        </View>
      ) : (
        <View style={styles.card}>
          <Text style={styles.body}>Evaluated: {report.evaluated_count}</Text>
          <Text style={styles.body}>Misclassified: {report.misclassified_count}</Text>
          <Text style={styles.body}>Low confidence: {report.low_confidence_count}</Text>
          <Text style={styles.body}>Shortcut cues: {report.shortcut_cue_count}</Text>
          {(report.failure_gallery || []).map((item) => (
            <View key={item.sample_id} style={styles.sampleCard}>
              <Text style={styles.cardTitle}>{item.sample_id}</Text>
              <Text style={styles.body}>
                Expected {item.true_label}; predicted {item.predicted_label} · confidence{" "}
                {(item.confidence * 100).toFixed(1)}%
              </Text>
              <Text style={styles.body}>{item.note}</Text>
            </View>
          ))}
        </View>
      )}
    </View>
  );
}

function LimitationsPanel() {
  return (
    <View>
      <Text style={styles.sectionTitle}>Scope and known limitations</Text>
      <Text style={styles.noticeText}>{NOTICE}</Text>
      <View style={styles.card}>
        <Text style={styles.body}>
          • This is an educational research prototype, not a medical device.{"\n"}
          • Do not use it for diagnosis, emergency triage, treatment decisions, or to delay care.
          {"\n"}
          • Confidence can be miscalibrated; a high score does not establish correctness.{"\n"}
          • Explanations are not causal evidence; segmentations are not clinical measurements.
          {"\n"}
          • PneumoniaMNIST is small, pediatric, downsampled, and explicitly not for clinical use.
          {"\n"}
          • Dataset, labeling, preprocessing, and population shifts can change model behavior.
        </Text>
      </View>
    </View>
  );
}

async function readJson(response) {
  if (typeof response?.ok === "boolean" && !response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return typeof response?.json === "function" ? response.json() : response;
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#f3f7f8" },
  header: { paddingHorizontal: 20, paddingTop: 14, paddingBottom: 12 },
  eyebrow: { color: "#087e8b", fontSize: 11, fontWeight: "700", letterSpacing: 1.2 },
  title: { color: "#102a43", fontSize: 30, fontWeight: "800", marginTop: 4 },
  subtitle: { color: "#526777", fontSize: 14, marginTop: 3 },
  notice: { backgroundColor: "#fff2d8", marginHorizontal: 16, padding: 12, borderRadius: 10 },
  noticeText: { color: "#815b0b", fontSize: 13, fontWeight: "700", lineHeight: 19 },
  connection: { color: "#587080", fontSize: 12, marginHorizontal: 20, marginTop: 9 },
  tabRow: { flexDirection: "row", gap: 7, paddingHorizontal: 14, paddingVertical: 12 },
  tab: { paddingVertical: 8, paddingHorizontal: 11, borderRadius: 20, backgroundColor: "#e3ebee" },
  activeTab: { backgroundColor: "#087e8b" },
  tabText: { color: "#35515f", fontSize: 13, fontWeight: "600" },
  activeTabText: { color: "#ffffff" },
  content: { paddingHorizontal: 16, paddingBottom: 36 },
  sectionTitle: { color: "#102a43", fontSize: 22, fontWeight: "800", marginBottom: 8 },
  body: { color: "#344e5c", fontSize: 14, lineHeight: 21, marginBottom: 8 },
  muted: { color: "#728591", fontSize: 13, lineHeight: 19, marginVertical: 5 },
  error: { color: "#9f2d2d", backgroundColor: "#fde8e7", padding: 12, borderRadius: 8 },
  spinner: { marginVertical: 12 },
  secondaryButton: {
    borderColor: "#087e8b",
    borderWidth: 1,
    borderRadius: 10,
    padding: 12,
    alignItems: "center",
    marginVertical: 8,
  },
  secondaryButtonText: { color: "#087e8b", fontWeight: "700" },
  primaryButton: {
    backgroundColor: "#087e8b",
    borderRadius: 10,
    padding: 14,
    alignItems: "center",
    marginVertical: 10,
  },
  disabledButton: { opacity: 0.55 },
  primaryButtonText: { color: "#ffffff", fontWeight: "700" },
  preview: { width: "100%", height: 260, backgroundColor: "#e8eff2", borderRadius: 12 },
  emptyImage: {
    height: 160,
    backgroundColor: "#e8eff2",
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
  },
  card: { backgroundColor: "#ffffff", padding: 14, borderRadius: 12, marginTop: 12 },
  cardTitle: { color: "#102a43", fontSize: 16, fontWeight: "700", marginTop: 8, marginBottom: 5 },
  overlay: { width: "100%", height: 220, backgroundColor: "#e8eff2", marginVertical: 8 },
  sampleCard: {
    backgroundColor: "#ffffff",
    padding: 10,
    borderRadius: 10,
    marginVertical: 6,
  },
  sampleImage: { width: "100%", height: 180, backgroundColor: "#e8eff2" },
});
