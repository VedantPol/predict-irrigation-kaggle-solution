async function loadData() {
  const response = await fetch("./data/latest_run.json", { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Failed to load dashboard data: ${response.status}`);
  }
  return response.json();
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(6);
}

function renderKpis(headline) {
  const grid = document.getElementById("kpiGrid");
  const cards = [
    { label: "Level4 Score", value: formatScore(headline.level4_valid_auc) },
    { label: "Stacking Score", value: formatScore(headline.stacking_valid_auc) },
    { label: "Models Trained", value: headline.models_trained ?? "n/a" },
    { label: "Stacking Models", value: headline.stacking_models ?? "n/a" },
    { label: "Hill Climb Score", value: formatScore(headline.hill_climb_valid_auc) },
    { label: "Pseudo Labels", value: headline.pseudo_rows ?? "n/a" },
    { label: "Pseudo Ratio", value: headline.pseudo_ratio ?? "n/a" },
    { label: "Train Rows", value: headline.extra_train_rows ?? "n/a" },
  ];

  grid.innerHTML = cards
    .map(
      (card) => `
      <article class="kpi">
        <p class="kpi-label">${card.label}</p>
        <p class="kpi-value">${card.value}</p>
      </article>
    `
    )
    .join("");
}

function renderMeta(meta) {
  const el = document.getElementById("runMeta");
  el.textContent = `Feature set: ${meta.feature_set_name} | Generated UTC: ${meta.generated_at_utc} | Source: ${meta.result_folder}`;
}

function renderTimelineChart(timeline) {
  const ctx = document.getElementById("timelineChart");
  const labels = timeline.map((row) => row.stage.toUpperCase());
  const values = timeline.map((row) => row.valid_auc ?? null);

  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Validation Score",
          data: values,
          borderColor: "#0d7a6f",
          backgroundColor: "rgba(13, 122, 111, 0.15)",
          fill: true,
          tension: 0.28,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: "#0b5c54",
        },
      ],
    },
    options: {
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          min: 0.2,
          max: 1.0,
          ticks: { callback: (v) => Number(v).toFixed(3) },
        },
      },
    },
  });
}

function renderLibraryChart(librarySummary) {
  const ctx = document.getElementById("libraryChart");
  const top = librarySummary.slice(0, 10);

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: top.map((row) => row.name.replace("dl_", "")),
      datasets: [
        {
          label: "Best Score",
          data: top.map((row) => row.best_valid_auc ?? 0),
          backgroundColor: top.map((row) =>
            row.kind === "deep" ? "rgba(13, 122, 111, 0.78)" : "rgba(217, 93, 57, 0.8)"
          ),
          borderRadius: 8,
        },
      ],
    },
    options: {
      indexAxis: "y",
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          min: 0.2,
          max: 1.0,
          ticks: { callback: (v) => Number(v).toFixed(3) },
        },
      },
    },
  });
}

function renderTopModelsTable(topModels) {
  const tbody = document.querySelector("#topModelsTable tbody");
  tbody.innerHTML = topModels
    .slice(0, 20)
    .map(
      (row) => `
      <tr>
        <td>${row.model_name}</td>
        <td>${row.library}</td>
        <td>${row.backend}</td>
        <td>${formatScore(row.valid_auc)}</td>
      </tr>
    `
    )
    .join("");
}

async function boot() {
  try {
    const data = await loadData();
    renderMeta(data.meta);
    renderKpis(data.headline);
    renderTimelineChart(data.timeline || []);
    renderLibraryChart(data.library_summary || []);
    renderTopModelsTable(data.top_models || []);
  } catch (error) {
    const meta = document.getElementById("runMeta");
    meta.textContent = `Unable to load dashboard data: ${error.message}`;
    console.error(error);
  }
}

boot();
