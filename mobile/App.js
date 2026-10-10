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
  TextInput,
  View,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";

const API_URL = (process.env.EXPO_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);
const NOTICE =
  "Demo only — not for diagnosis, treatment decisions, or emergencies.";
const TABS = [
  ["cases", "Demo Cases"],
  ["review", "Workspace"],
  ["report", "Report"],
  ["failures", "Failures"],
  ["about", "Limitations"],
];
const DEMO_CASE_NAMES = [
  {
    name: "Avery Chen",
    label: "normal",
    sampleOrdinal: 1,
    displayId: "XR-482731",
    scanDateTime: "2025-04-12 · 09:42 AM",
  },
  {
    name: "Jordan Rivera",
    label: "pneumonia",
    sampleOrdinal: 1,
    displayId: "XR-195604",
    scanDateTime: "2025-05-03 · 02:18 PM",
  },
  {
    name: "Taylor Morgan",
    label: "normal",
    sampleOrdinal: 2,
    displayId: "XR-730816",
    scanDateTime: "2025-06-21 · 11:07 AM",
  },
  {
    name: "Riley Patel",
    label: "pneumonia",
    sampleOrdinal: 2,
    displayId: "XR-361952",
    scanDateTime: "2025-07-09 · 08:35 AM",
  },
  {
    name: "Casey Brooks",
    label: "normal",
    sampleOrdinal: 3,
    displayId: "XR-847203",
    scanDateTime: "2025-08-16 · 04:51 PM",
  },
  {
    name: "Quinn Parker",
    label: "pneumonia",
    sampleOrdinal: 3,
    displayId: "XR-529174",
    scanDateTime: "2025-09-02 · 10:26 AM",
  },
];

export default function App() {
  const [tab, setTab] = useState("cases");
  const [serverStatus, setServerStatus] = useState("Connecting to demo service…");
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [examples, setExamples] = useState([]);
  const [selectedExample, setSelectedExample] = useState(0);
  const [caseSearch, setCaseSearch] = useState("");
  const [examplesError, setExamplesError] = useState("");
  const [failures, setFailures] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [wideLayout, setWideLayout] = useState(false);

  useEffect(() => {
    if (Platform.OS !== "web") return undefined;
    const updateLayout = () => setWideLayout(window.innerWidth >= 900);
    updateLayout();
    window.addEventListener("resize", updateLayout);
    return () => window.removeEventListener("resize", updateLayout);
  }, []);

  useEffect(() => {
    loadExamples();
    fetch(`${API_URL}/api/health`)
      .then(readJson)
      .then((health) => {
        if (health.purpose !== "demo") {
          setServerStatus("API is outdated · restart the Python server from this project");
          return;
        }
        setServerStatus(
          `Demo API connected · ${Object.values(health.features || {}).filter(Boolean).length} model components configured`,
        );
      })
      .catch(() => setServerStatus("API unavailable · start the Python server on this network"));
  }, []);

  useEffect(() => {
    if (Platform.OS !== "web") return undefined;

    function handleDragOver(event) {
      if (!event.dataTransfer?.types?.includes("Files")) return;
      event.preventDefault();
      setIsDragging(true);
    }

    function handleDragLeave(event) {
      if (!event.relatedTarget) setIsDragging(false);
    }

    function handleDrop(event) {
      if (!event.dataTransfer?.files?.length) return;
      event.preventDefault();
      setIsDragging(false);
      const file = event.dataTransfer.files[0];
      if (!file.type.startsWith("image/")) {
        setError("Choose an image file to preview and analyze.");
        return;
      }
      setImage({
        uri: URL.createObjectURL(file),
        fileName: file.name,
        mimeType: file.type,
        webFile: file,
      });
      setResult(null);
      setError("");
      setTab("review");
    }

    window.addEventListener("dragover", handleDragOver);
    window.addEventListener("dragleave", handleDragLeave);
    window.addEventListener("drop", handleDrop);
    return () => {
      window.removeEventListener("dragover", handleDragOver);
      window.removeEventListener("dragleave", handleDragLeave);
      window.removeEventListener("drop", handleDrop);
    };
  }, []);

  async function chooseImage() {
    try {
      setError("");
      if (Platform.OS === "web") {
        const selection = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ["images"],
          quality: 1,
          allowsEditing: false,
        });
        if (selection.canceled || !selection.assets?.length) return;
        setImage(selection.assets[0]);
        setResult(null);
        return;
      }
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
    if (!image && !examples[selectedExample]) {
      setError("Wait for a benchmark image to load or upload an image.");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);
    try {
      if (!image && examples[selectedExample]) {
        const sampleId = examples[selectedExample].sample_id;
        const response = await fetch(`${API_URL}/api/analyze-sample/${sampleId}`, {
          method: "POST",
        });
        setResult(await readJson(response));
        return;
      }
      const body = new FormData();
      const filename = image.fileName || "medscan-demo-sample.jpg";
      if (Platform.OS === "web") {
        if (image.webFile) {
          body.append("file", image.webFile, filename);
        } else {
          const response = await fetch(image.uri);
          if (!response.ok) {
            throw new Error("Could not read the selected image from the browser.");
          }
          body.append("file", await response.blob(), filename);
        }
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
    setExamplesError("");
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/samples?limit=6&label=all`);
      const payload = await readJson(response);
      if (
        !Array.isArray(payload.examples) ||
        payload.examples.some((example) => !example.sample_id || !example.image_url)
      ) {
        throw new Error("The Python API is outdated. Restart it from the latest project files.");
      }
      const bySampleId = new Map(
        (payload.examples || []).map((example) => [example.sample_id, example]),
      );
      const assignedExamples = DEMO_CASE_NAMES.map((demoCase) => {
        const sampleId =
          `PMN-TRAIN-${demoCase.label.toUpperCase()}-` +
          String(demoCase.sampleOrdinal).padStart(4, "0");
        return bySampleId.get(sampleId) || null;
      });
      if (assignedExamples.some((example) => !example)) {
        throw new Error("The API did not return all six assigned benchmark cases.");
      }
      setExamples(assignedExamples);
      setSelectedExample(0);
    } catch (requestError) {
      setExamplesError(
        `Could not load benchmark samples from ${API_URL}/api/samples. ${requestError.message} Check that the Python API is running and EXPO_PUBLIC_API_URL points to the computer running it. The first download may take a few minutes.`,
      );
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
    if (nextTab === "failures" && failures === null) loadFailures();
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <View style={styles.header}>
        <View>
          <Text style={styles.eyebrow}>IMAGING DEMO WORKSPACE</Text>
          <Text style={styles.title}>MedScan</Text>
        </View>
        <View style={styles.headerBadge}>
          <Text style={styles.headerBadgeText}>DEMO ONLY</Text>
        </View>
      </View>

      <View style={styles.notice}>
        <Text style={styles.noticeText}>{NOTICE}</Text>
      </View>
      <Text style={styles.connection}>{serverStatus}</Text>

      {isDragging ? (
        <View pointerEvents="none" style={styles.dropBanner}>
          <Text style={styles.dropBannerText}>Drop the scan image to add it to this demo</Text>
        </View>
      ) : null}

      <View style={[styles.mainLayout, wideLayout && styles.wideMainLayout]}>
        <ScrollView
          style={wideLayout ? styles.sideNavigation : undefined}
          horizontal={!wideLayout}
          showsHorizontalScrollIndicator={false}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={[styles.tabRow, wideLayout && styles.sideTabRow]}
        >
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
        </ScrollView>

        <ScrollView style={styles.mainContent} contentContainerStyle={styles.content}>
          {error ? <Text style={styles.error}>{error}</Text> : null}
          {busy ? <ActivityIndicator color="#087e8b" style={styles.spinner} /> : null}

          {tab === "review" || tab === "cases" ? (
            <WorkspacePanel
              image={image}
              result={result}
              examples={examples}
              selectedIndex={selectedExample}
              samplesError={examplesError}
              busy={busy}
              search={caseSearch}
              onSearch={setCaseSearch}
              onSelectNamedCase={(index) => {
                setSelectedExample(index);
                setCaseSearch("");
                setImage(null);
                setResult(null);
                setError("");
              }}
              onChoose={chooseImage}
              onAnalyze={analyzeImage}
              onReload={loadExamples}
            />
          ) : null}
          {tab === "report" ? (
            <ReportPanel
              result={result}
              image={image}
              selectedSample={examples[selectedExample]}
              selectedName={DEMO_CASE_NAMES[selectedExample]?.name}
            />
          ) : null}
          {tab === "failures" ? <FailuresPanel report={failures} /> : null}
          {tab === "about" ? <LimitationsPanel /> : null}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

function WorkspacePanel({
  image,
  result,
  examples,
  selectedIndex,
  samplesError,
  onChoose,
  onAnalyze,
  onReload,
  search,
  onSearch,
  onSelectNamedCase,
}) {
  const selectedSample = examples[selectedIndex];
  const selectedCase = DEMO_CASE_NAMES[selectedIndex];
  const caseSearchActive = search.trim().length > 0;
  const caseQuery = search.trim().toLowerCase();
  const [previewError, setPreviewError] = useState(false);
  const activeImage = image
    ? { uri: image.uri }
    : selectedSample
      ? { uri: `${API_URL}${selectedSample.image_url}` }
      : null;
  useEffect(() => setPreviewError(false), [activeImage?.uri]);

  return (
    <View>
      <Text style={styles.sectionTitle}>Case workspace</Text>
      <Text style={styles.body}>
        Search for a fictional demo name to load its assigned public benchmark image, or upload
        your own image.
      </Text>
      <Text style={styles.muted}>
        On a computer, drag an image anywhere onto this page, or use Upload to choose a file.
        Searching for a name shows only matching cases.
      </Text>

      {examples.length && !image ? (
        <View style={styles.casePicker}>
          <Text style={styles.cardTitle}>Find a demo case by name</Text>
          <TextInput
            accessibilityLabel="Search fictional demo names"
            placeholder="Search fictional name"
            placeholderTextColor="#647887"
            value={search}
            onChangeText={onSearch}
            style={styles.searchInput}
          />
          {caseSearchActive
            ? ["normal", "pneumonia"].map((category) => {
                const categoryCases = DEMO_CASE_NAMES
                  .map((demoCase, index) => ({ ...demoCase, index }))
                  .filter(
                    (demoCase) =>
                      demoCase.label === category &&
                      demoCase.name.toLowerCase().includes(caseQuery),
                  );
                if (!categoryCases.length) return null;
                return (
                  <View key={category}>
                    <Text style={styles.caseCategoryHeading}>
                      {category === "normal" ? "Normal" : "Pneumonia"} dataset-label category
                    </Text>
                    {categoryCases.map((demoCase) => (
                      <Pressable
                        accessibilityRole="button"
                        accessibilityLabel={demoCase.name}
                        accessibilityState={{ selected: selectedIndex === demoCase.index }}
                        key={demoCase.name}
                        onPress={() => onSelectNamedCase(demoCase.index)}
                        style={[
                          styles.demoCaseRow,
                          selectedIndex === demoCase.index && styles.selectedDemoCase,
                        ]}
                      >
                        <View style={styles.demoCaseHeading}>
                          <Text style={styles.chartName}>{demoCase.name}</Text>
                          <Text style={styles.recordStatus}>FICTIONAL</Text>
                        </View>
                      </Pressable>
                    ))}
                  </View>
                );
              })
            : null}
          {caseSearchActive &&
          !DEMO_CASE_NAMES.some((demoCase) =>
            demoCase.name.toLowerCase().includes(caseQuery),
          ) ? (
            <Text style={styles.muted}>No fictional demo name matches that search.</Text>
          ) : null}
        </View>
      ) : null}
      {!examples.length ? (
        <View>
          <Text style={styles.muted}>
            Six benchmark images load from the Python API on startup.
          </Text>
          {samplesError ? <Text style={styles.error}>{samplesError}</Text> : null}
          <Pressable onPress={onReload} style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>Retry loading samples</Text>
          </Pressable>
        </View>
      ) : null}

      <View style={styles.workspaceGrid}>
        <View style={styles.recordCard}>
          <View style={styles.recordHeader}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>
                {image ? "U" : selectedCase?.name?.[0] || "D"}
              </Text>
            </View>
            <View style={styles.recordHeading}>
              <Text style={styles.chartName}>
                {image ? "Uploaded image" : selectedCase?.name || "Demo case"}
              </Text>
              <Text style={styles.recordSubtitle}>
                {image
                  ? "User-selected image · no patient record"
                  : "Fictional demo name · no patient record"}
              </Text>
            </View>
            {!image ? <Text style={styles.recordStatus}>SIMULATED</Text> : null}
          </View>
          <View style={styles.recordDetails}>
            <ChartField
              label="Dataset category"
              value={
                image
                  ? "Unavailable for uploads"
                  : selectedSample?.label
                      ?.replace("Benchmark label: ", "")
                      .replace(" (not a diagnosis)", "") || "Unavailable"
              }
            />
            {!image ? (
              <>
                <ChartField label="Demo scan ID" value={selectedCase?.displayId || "Not assigned"} />
                <ChartField
                  label="Demo scan date and time (simulated)"
                  value={selectedCase?.scanDateTime || "Not available"}
                />
              </>
            ) : null}
            <ChartField label="Diagnosis" value="Not provided by this demo" />
          </View>
          <Text style={styles.syntheticNote}>
            {image
              ? "Uploaded image only. No patient identity or scan date/time is inferred."
              : "Fictional profile and scan date/time for this interface only. The assigned public benchmark image is not this person's scan and carries no patient acquisition time."}
          </Text>
        </View>

        <View style={styles.imageWorkspace}>
          <View style={styles.imageHeader}>
            <View>
              <Text style={styles.cardTitle}>Imaging sample</Text>
              <Text style={styles.recordSubtitle}>
                {image
                  ? image.fileName || "Uploaded image"
                  : selectedCase?.name || "Loading demo cases…"}
              </Text>
            </View>
            <Pressable onPress={onChoose} style={styles.uploadButton}>
              <Text style={styles.uploadButtonText}>＋ Upload</Text>
            </Pressable>
            {image ? (
              <Pressable
                onPress={() => onSelectNamedCase(selectedIndex)}
                style={styles.uploadButton}
              >
                <Text style={styles.uploadButtonText}>Return to demo case</Text>
              </Pressable>
            ) : null}
          </View>

          {activeImage && !previewError ? (
            <Image
              source={activeImage}
              style={styles.workspaceImage}
              resizeMode="contain"
              accessibilityLabel="Selected public benchmark or uploaded image"
              onError={() => setPreviewError(true)}
            />
          ) : (
            <View style={[styles.workspaceImage, styles.emptyImage]}>
              <Text style={styles.muted}>
                {previewError
                  ? "Could not load the image. Check that the Python API is reachable, then reload demo cases."
                  : "Benchmark image is not available yet."}
              </Text>
            </View>
          )}

          <Text style={styles.imageSourceNote}>
            {image
              ? "User-selected image · not a patient record"
              : `Public pediatric chest X-ray benchmark · demo assignment for ${selectedCase?.name || "fictional case"} · not a patient scan`}
          </Text>

          <Text style={styles.datasetLabel}>
            {image
              ? "Uploaded demo image"
              : `Dataset reference category: ${selectedCase?.label || "unavailable"}`}
          </Text>
          <Text style={styles.muted}>
            Categories are dataset annotations, not patient diagnoses or clinical findings.
          </Text>

          <Pressable
            accessibilityRole="button"
            onPress={onAnalyze}
            style={styles.primaryButton}
          >
            <Text style={styles.primaryButtonText}>Run demo analysis</Text>
          </Pressable>
          {result ? <PredictionSummary result={result} /> : null}
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Model attribution intensity</Text>
        <Text style={styles.body}>
          Highlighted regions show pixel influence for the low-resolution linear demo classifier.
          They do not show disease location, severity, or probability.
        </Text>
        {result?.explanation_overlay ? (
          <>
            <Image
              source={{ uri: result.explanation_overlay }}
              style={styles.overlay}
              resizeMode="contain"
              accessibilityLabel="Model attribution heatmap overlay"
            />
            <View style={styles.intensityLegend}>
              <Text style={styles.muted}>Lower attribution</Text>
              <View style={styles.legendGradient} />
              <Text style={styles.muted}>Higher attribution</Text>
            </View>
          </>
        ) : (
          <View style={styles.emptyOverlay}>
            <Text style={styles.muted}>
              {result
                ? "Attribution overlay is unavailable until an explanation model is connected."
                : "Run demo analysis to check whether an attribution overlay is available."}
            </Text>
          </View>
        )}
      </View>
    </View>
  );
}

function PredictionSummary({ result }) {
  const prediction = result.prediction;
  return (
    <View style={[styles.card, prediction ? styles.predictionCard : styles.unavailableCard]}>
      <Text style={styles.cardTitle}>
        {prediction
          ? `Benchmark class estimate: ${prediction.label}`
          : "Prediction unavailable"}
      </Text>
      {prediction ? (
        <Text style={styles.body}>
          Top-class model score: {(result.confidence * 100).toFixed(1)}%
        </Text>
      ) : (
        <Text style={styles.body}>No classifier is connected; no prediction was made.</Text>
      )}
      <Text style={styles.muted}>
        {result.screening?.status
          ? `Upload screening: ${result.screening.status}`
          : "Upload screening unavailable without a model prediction."}
      </Text>
    </View>
  );
}

function ReportPanel({ result, image, selectedSample, selectedName }) {
  if (!result) {
    return (
      <View>
        <Text style={styles.sectionTitle}>Analysis report</Text>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>No analysis yet</Text>
          <Text style={styles.body}>
            Select a benchmark or uploaded image in the Workspace and run the demo analysis.
          </Text>
        </View>
      </View>
    );
  }

  const prediction = result.prediction;
  return (
    <View>
      <Text style={styles.sectionTitle}>Analysis report</Text>
      <Text style={styles.body}>
        {image?.fileName
          ? `Image: ${image.fileName}`
          : selectedName || result.sample?.sample_id || selectedSample?.sample_id || "Demo image"}
      </Text>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Prediction</Text>
        {prediction ? (
          <>
            <Text style={styles.reportLabel}>{prediction.label}</Text>
            <Text style={styles.body}>
              Top-class model score: {(result.confidence * 100).toFixed(1)}%
            </Text>
            <Text style={styles.muted}>
              These are scores for benchmark dataset classes, not calibrated probabilities of a
              medical condition.
            </Text>
            <Text style={styles.cardTitle}>Benchmark class scores</Text>
            {Object.entries(prediction.probs).map(([label, probability]) => (
              <ProbabilityBar key={label} label={label} probability={probability} />
            ))}
          </>
        ) : (
          <Text style={styles.body}>
            No classifier is connected. The absence of a prediction is not a normal or negative
            result.
          </Text>
        )}
      </View>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Upload screening</Text>
        <Text style={styles.body}>{result.screening?.status || "Unavailable"}</Text>
        {result.screening?.reasons?.length ? (
          <Text style={styles.body}>Notes: {result.screening.reasons.join(", ")}</Text>
        ) : null}
        <Text style={styles.muted}>
          Screening heuristics cannot establish that an image is supported or safe to assess.
        </Text>
      </View>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Model explanation</Text>
        {result.explanation_overlay ? (
          <>
            <Image
              source={{ uri: result.explanation_overlay }}
              style={styles.overlay}
              resizeMode="contain"
            />
            <Text style={styles.muted}>
              This low-resolution linear-feature attribution reflects model influence only—not
              disease location, severity, or probability.
            </Text>
          </>
        ) : (
          <Text style={styles.muted}>No explanation overlay was returned.</Text>
        )}
      </View>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Segmentation</Text>
        {result.segmentation_overlay ? (
          <Image source={{ uri: result.segmentation_overlay }} style={styles.overlay} resizeMode="contain" />
        ) : (
          <Text style={styles.muted}>No segmentation overlay was returned.</Text>
        )}
        <Text style={styles.muted}>Segmentation output is not a clinical measurement.</Text>
      </View>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Labeled-example comparison</Text>
        {result.comparison ? (
          <>
            <Text style={styles.body}>{result.comparison.label}</Text>
            <Image source={{ uri: result.comparison.image }} style={styles.overlay} resizeMode="contain" />
          </>
        ) : (
          <Text style={styles.muted}>No labeled example comparison was returned.</Text>
        )}
      </View>
      <Text style={styles.reportNotice}>{NOTICE}</Text>
    </View>
  );
}

function ProbabilityBar({ label, probability }) {
  const percent = Math.max(0, Math.min(1, probability)) * 100;
  return (
    <View style={styles.probabilityRow}>
      <View style={styles.probabilityLabels}>
        <Text style={styles.body}>{label}</Text>
        <Text style={styles.body}>{percent.toFixed(1)}%</Text>
      </View>
      <View style={styles.probabilityTrack}>
        <View style={[styles.probabilityFill, { width: `${percent}%` }]} />
      </View>
    </View>
  );
}

function ChartField({ label, value }) {
  return (
    <View style={styles.chartField}>
      <Text style={styles.chartFieldLabel}>{label}</Text>
      <Text style={styles.chartFieldValue}>{value}</Text>
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
          • This is an educational demonstration, not a medical device.{"\n"}
          • Do not use it for diagnosis, emergency triage, treatment decisions, or to delay care.
          {"\n"}
          • Confidence can be miscalibrated; a high score does not establish correctness.{"\n"}
          • The bundled classifier is a simple low-resolution benchmark baseline; scores are not
          calibrated medical probabilities.{"\n"}
          • Explanations are not causal evidence; segmentations are not clinical measurements.
          {"\n"}
          • PneumoniaMNIST is small, pediatric, downsampled, and explicitly not for clinical use.
          {"\n"}
          • The synthetic demo record is not linked to uploaded images.
          {"\n"}
          • Dataset, labeling, preprocessing, and population shifts can change model behavior.
          {"\n"}
          • Upload screening is heuristic and cannot guarantee that an image is a supported X-ray.
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
  safeArea: { flex: 1, backgroundColor: "#eaf4fc" },
  header: {
    backgroundColor: "#d9edfb",
    borderBottomColor: "#c4dff2",
    borderBottomWidth: 1,
    paddingHorizontal: 24,
    paddingTop: 18,
    paddingBottom: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  eyebrow: { color: "#31566f", fontSize: 11, fontWeight: "700", letterSpacing: 1.35 },
  title: { color: "#111820", fontSize: 30, fontWeight: "800", marginTop: 4 },
  headerBadge: {
    backgroundColor: "#ffffff",
    borderColor: "#b9d7ed",
    borderWidth: 1,
    paddingHorizontal: 11,
    paddingVertical: 7,
    borderRadius: 18,
  },
  headerBadgeText: { color: "#31566f", fontSize: 10, fontWeight: "800", letterSpacing: 0.6 },
  notice: {
    backgroundColor: "#fff7e4",
    borderColor: "#eed99b",
    borderWidth: 1,
    marginHorizontal: 20,
    marginTop: 14,
    padding: 11,
    borderRadius: 9,
  },
  noticeText: { color: "#57400a", fontSize: 12, fontWeight: "700", lineHeight: 18 },
  connection: { color: "#485b69", fontSize: 12, marginHorizontal: 24, marginTop: 9 },
  tabRow: { flexDirection: "row", gap: 7, paddingHorizontal: 14, paddingVertical: 12 },
  mainLayout: { flex: 1, flexDirection: "column" },
  wideMainLayout: { flexDirection: "row" },
  mainContent: { flex: 1, width: "100%" },
  sideNavigation: {
    width: 184,
    flexGrow: 0,
    borderRightWidth: 1,
    borderRightColor: "#c9dfef",
  },
  sideTabRow: {
    flexDirection: "column",
    alignItems: "stretch",
    width: 184,
    paddingHorizontal: 12,
    paddingTop: 18,
  },
  dropBanner: {
    position: "absolute",
    zIndex: 20,
    top: 18,
    left: 18,
    right: 18,
    borderWidth: 2,
    borderStyle: "dashed",
    borderColor: "#326f98",
    backgroundColor: "#eaf4fc",
    borderRadius: 12,
    padding: 18,
    alignItems: "center",
  },
  dropBannerText: { color: "#234d68", fontWeight: "800", fontSize: 15 },
  tab: {
    paddingVertical: 9,
    paddingHorizontal: 14,
    borderRadius: 8,
    backgroundColor: "#dcecf8",
    borderWidth: 1,
    borderColor: "#c9dfef",
  },
  activeTab: { backgroundColor: "#ffffff", borderColor: "#76a9cd" },
  tabText: { color: "#354755", fontSize: 13, fontWeight: "600" },
  activeTabText: { color: "#17232b" },
  content: { paddingHorizontal: 20, paddingTop: 8, paddingBottom: 40, maxWidth: 1320, width: "100%", alignSelf: "center" },
  sectionTitle: { color: "#111820", fontSize: 23, fontWeight: "800", marginBottom: 8 },
  body: { color: "#202b33", fontSize: 14, lineHeight: 21, marginBottom: 8 },
  muted: { color: "#566875", fontSize: 13, lineHeight: 19, marginVertical: 5 },
  error: { color: "#9f2d2d", backgroundColor: "#fde8e7", padding: 12, borderRadius: 8 },
  errorCard: { backgroundColor: "#fff1f0", borderRadius: 10, padding: 10, marginVertical: 8 },
  spinner: { marginVertical: 12 },
  secondaryButton: {
    borderColor: "#6e9fbe",
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    alignItems: "center",
    marginVertical: 8,
    backgroundColor: "#f5faff",
  },
  secondaryButtonText: { color: "#234d68", fontWeight: "700" },
  primaryButton: {
    backgroundColor: "#326f98",
    borderRadius: 8,
    padding: 14,
    alignItems: "center",
    marginVertical: 10,
  },
  disabledButton: { opacity: 0.55 },
  primaryButtonText: { color: "#ffffff", fontWeight: "700", fontSize: 14 },
  preview: { width: "100%", height: 260, backgroundColor: "#e3f0fa", borderRadius: 8 },
  emptyImage: {
    height: 160,
    backgroundColor: "#e3f0fa",
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
  },
  card: {
    backgroundColor: "#ffffff",
    borderColor: "#d1e2ee",
    borderWidth: 1,
    padding: 16,
    borderRadius: 10,
    marginTop: 12,
  },
  workspaceGrid: { flexDirection: "row", flexWrap: "wrap", alignItems: "flex-start", gap: 14 },
  recordCard: {
    flexGrow: 1,
    flexBasis: 270,
    backgroundColor: "#ffffff",
    borderColor: "#d1e2ee",
    borderWidth: 1,
    borderRadius: 10,
    padding: 16,
    marginTop: 12,
  },
  imageWorkspace: {
    flexGrow: 2,
    flexBasis: 440,
    backgroundColor: "#ffffff",
    borderColor: "#d1e2ee",
    borderWidth: 1,
    borderRadius: 10,
    padding: 16,
    marginTop: 12,
  },
  recordHeader: { flexDirection: "row", alignItems: "center", gap: 10 },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "#dceefa",
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: { color: "#284d67", fontSize: 17, fontWeight: "800" },
  recordHeading: { flex: 1 },
  recordSubtitle: { color: "#566875", fontSize: 12, lineHeight: 18 },
  recordStatus: {
    color: "#6b5210",
    backgroundColor: "#fff2d5",
    paddingHorizontal: 8,
    paddingVertical: 5,
    borderRadius: 5,
    fontSize: 9,
    fontWeight: "800",
  },
  recordDetails: { marginTop: 12 },
  syntheticNote: {
    color: "#654d13",
    backgroundColor: "#fff7e4",
    borderRadius: 6,
    padding: 9,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 8,
  },
  imageHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
    marginBottom: 8,
  },
  uploadButton: {
    backgroundColor: "#eaf4fc",
    borderColor: "#b9d7ed",
    borderWidth: 1,
    borderRadius: 7,
    paddingVertical: 8,
    paddingHorizontal: 11,
  },
  uploadButtonText: { color: "#234d68", fontWeight: "700", fontSize: 12 },
  workspaceImage: {
    width: "100%",
    height: 340,
    backgroundColor: "#edf5fb",
    borderRadius: 8,
  },
  imageSourceNote: {
    color: "#475d6b",
    fontSize: 12,
    lineHeight: 18,
    marginTop: 8,
  },
  datasetLabel: { color: "#151d24", fontSize: 14, fontWeight: "700", marginTop: 5 },
  predictionCard: { borderLeftWidth: 4, borderLeftColor: "#568bae" },
  unavailableCard: { borderLeftWidth: 4, borderLeftColor: "#bd8c35" },
  cardTitle: { color: "#151d24", fontSize: 16, fontWeight: "700", marginTop: 8, marginBottom: 5 },
  reportLabel: { color: "#234d68", fontSize: 25, fontWeight: "800", marginBottom: 6 },
  reportNotice: {
    color: "#815b0b",
    backgroundColor: "#fff2d8",
    fontWeight: "700",
    lineHeight: 20,
    padding: 12,
    borderRadius: 10,
    marginTop: 12,
  },
  overlay: { width: "100%", height: 220, backgroundColor: "#edf5fb", marginVertical: 8 },
  emptyOverlay: {
    minHeight: 120,
    padding: 16,
    backgroundColor: "#edf5fb",
    borderRadius: 10,
    justifyContent: "center",
  },
  intensityLegend: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
    marginTop: 4,
  },
  legendGradient: {
    flex: 1,
    height: 10,
    borderRadius: 5,
    backgroundColor: "#e9bd55",
    borderWidth: 1,
    borderColor: "#c98e42",
  },
  probabilityRow: { marginBottom: 8 },
  probabilityLabels: { flexDirection: "row", justifyContent: "space-between" },
  probabilityTrack: {
    height: 9,
    backgroundColor: "#dcecf8",
    borderRadius: 5,
    overflow: "hidden",
  },
  probabilityFill: { height: "100%", backgroundColor: "#568bae", borderRadius: 5 },
  sampleCard: {
    backgroundColor: "#ffffff",
    padding: 10,
    borderRadius: 10,
    marginVertical: 6,
  },
  syntheticNotice: {
    backgroundColor: "#fff2d8",
    borderColor: "#e3c87a",
    borderWidth: 1,
    borderRadius: 12,
    padding: 14,
    marginVertical: 8,
  },
  syntheticNoticeTitle: {
    color: "#815b0b",
    fontSize: 13,
    fontWeight: "800",
    marginBottom: 6,
  },
  syntheticNoticeText: { color: "#684f1d", fontSize: 13, lineHeight: 19 },
  searchInput: {
    backgroundColor: "#ffffff",
    borderColor: "#b9d7ed",
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 13,
    paddingVertical: 11,
    color: "#17232b",
    fontSize: 14,
    marginVertical: 10,
  },
  caseCategoryHeading: {
    color: "#31566f",
    fontSize: 13,
    fontWeight: "800",
    marginTop: 12,
    marginBottom: 3,
  },
  demoCaseRow: {
    backgroundColor: "#ffffff",
    borderColor: "#d1e2ee",
    borderWidth: 1,
    borderRadius: 10,
    padding: 14,
    marginTop: 9,
  },
  selectedDemoCase: { borderColor: "#568bae", borderWidth: 2 },
  demoCaseHeading: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  chartName: { color: "#151d24", fontSize: 16, fontWeight: "800", marginBottom: 3 },
  chartField: {
    borderBottomColor: "#e7eef1",
    borderBottomWidth: 1,
    paddingVertical: 9,
  },
  chartFieldLabel: { color: "#566875", fontSize: 12, fontWeight: "600", marginBottom: 3 },
  chartFieldValue: { color: "#17232b", fontSize: 14, lineHeight: 20 },
});
