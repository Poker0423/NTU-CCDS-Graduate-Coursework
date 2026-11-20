(() => {
  "use strict";

  const DATA = window.DASHBOARD_DATA;
  const COLORS = {
    bg: "#090b10",
    surface: "#11151d",
    line: "#2b3340",
    text: "#f5f7fa",
    muted: "#98a1b2",
    red: "#ff3b30",
    blue: "#31c3ff",
    gold: "#ffd166",
    silver: "#c7d0d9",
    bronze: "#d99559"
  };
  const CONFIG = { displayModeBar: false, responsive: true, scrollZoom: false };
  const state = {
    era: "2010-2024",
    version: "reviewed",
    driver: "all",
    circuit: "all",
    gridBand: null,
    metric: "win",
    winsOnly: false,
    selectedEvent: null,
    activeTab: "overview",
    replayId: null,
    replayLap: 0,
    replayTimer: null
  };

  const sourceData = () => DATA.datasets[state.version];
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const integer = new Intl.NumberFormat("en-US");
  const percent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;

  function eraMatch(year) {
    if (state.era === "all") return true;
    const [start, end] = state.era.split("-").map(Number);
    return year >= start && year <= end;
  }

  function bandFor(grid) {
    if (grid === 1) return "P1";
    if (grid <= 5) return "P2–P5";
    if (grid <= 10) return "P6–P10";
    if (grid <= 15) return "P11–P15";
    if (grid <= 20) return "P16–P20";
    return "P21+";
  }

  function baseFilter(row) {
    return eraMatch(row.y)
      && (state.driver === "all" || row.d === state.driver)
      && (state.circuit === "all" || row.c === state.circuit);
  }

  function filteredRecords(includeGridBand = true) {
    return sourceData().records.filter((row) => baseFilter(row) && (!includeGridBand || !state.gridBand || bandFor(row.g) === state.gridBand));
  }

  function filteredEvents() {
    return sourceData().events.filter((row) => baseFilter(row)
      && (!state.gridBand || bandFor(row.g) === state.gridBand)
      && (!state.winsOnly || row.f === 1));
  }

  function populateFilters() {
    const drivers = new Map();
    const circuits = new Map();
    sourceData().records.forEach((row) => {
      if (!drivers.has(row.d)) drivers.set(row.d, row.dn);
      if (!circuits.has(row.c)) circuits.set(row.c, row.cn);
    });
    const driverSelect = $("#driverFilter");
    [...drivers.entries()].sort((a, b) => a[1].localeCompare(b[1])).forEach(([value, label]) => {
      driverSelect.add(new Option(label, value));
    });
    const circuitSelect = $("#circuitFilter");
    [...circuits.entries()].sort((a, b) => a[1].localeCompare(b[1])).forEach(([value, label]) => {
      circuitSelect.add(new Option(label, value));
    });
  }

  function plotLayout(overrides = {}) {
    return {
      paper_bgcolor: COLORS.surface,
      plot_bgcolor: COLORS.surface,
      font: { family: "Inter, Segoe UI, Arial, sans-serif", color: COLORS.text, size: 14 },
      margin: { l: 55, r: 18, t: 8, b: 48 },
      hoverlabel: { bgcolor: "#171c26", bordercolor: COLORS.line, font: { color: COLORS.text, size: 13 } },
      ...overrides
    };
  }

  function renderKpis() {
    const rows = filteredRecords();
    const events = rows.filter(row=>row.vf&&row.g>=11&&row.f!==null&&row.f<=3);
    const p11 = rows.filter((row) => row.g >= 11);
    const p11Podiums = p11.filter((row) => row.f !== null && row.f <= 3).length;
    const p17 = rows.filter((row) => row.g === 17);
    const p17Wins = p17.filter((row) => row.f === 1).length;

    $("#kpiEntries").textContent = integer.format(rows.length);
    $("#kpiEvents").textContent = integer.format(events.length);
    $("#kpiRate").textContent = p11.length ? percent(p11Podiums / p11.length, 2) : "n.a.";
    $("#kpiP17").textContent = p17.length ? `${p17Wins} / ${integer.format(p17.length)}` : "n.a.";
    const focus = state.gridBand ? ` · ${state.gridBand}` : "";
    $("#sampleLabel").textContent = `${integer.format(rows.length)} entries${focus}`;
    $(".period-label").textContent=state.era==="all"?"1950–2024":state.era.replace("-","–");
    $("#scopeNote").textContent=`${state.era==="all"?"Historical extension: 1950–2024":"Active sample: "+state.era.replace("-","–")} · ${state.version==="reviewed"?"3 sourced grid corrections":"Original grid values"} · Grid 0 excluded; withdrawals retained. Counts are entries, not confirmed starters.​`;
  }

  function renderHeatmap() {
    const rows = filteredRecords();
    const size = 30;
    const counts = Array.from({ length: size + 1 }, () => Array(size).fill(0));
    const denominators = Array(size).fill(0);
    const outcome = (row) => row.f === null ? size : Math.min(size, row.f) - 1;
    rows.forEach((row) => {
      const x = Math.min(size, row.g) - 1;
      if (x < 0) return;
      counts[outcome(row)][x] += 1; denominators[x] += 1;
    });
    const z = counts.map((row) => row.map((n, x) => denominators[x] ? n / denominators[x] : null));
    const customdata = counts.map((row) => row.map((n,x) => [n, denominators[x]]));
    const traces = [{type:'heatmap',x:Array.from({length:size},(_,i)=>i+1),y:Array.from({length:size+1},(_,i)=>i+1),
      z,customdata,zmin:0,zmax:1,zsmooth:false,
      colorscale:[[0,COLORS.bg],[.15,'#12324a'],[.5,'#168fbd'],[1,'#b7f0ff']],
      colorbar:{title:{text:'Share',side:'right'},tickformat:'.0%',thickness:10,len:.72},
      hovertemplate:'Recorded grid %{x}<br>Outcome row %{y}<br>%{customdata[0]} / %{customdata[1]} entries<br>%{z:.1%}<extra>Row 31 = unclassified; row 30 = P30+</extra>'}];
    Plotly.react('heatmapChart',traces,plotLayout({
      xaxis:{title:'Recorded grid position',range:[.5,30.5],tickvals:[1,5,10,15,20,25,30],ticktext:['1','5','10','15','20','25','30+'],zeroline:false},
      yaxis:{title:'Final classification',range:[31.6,.5],tickvals:[1,5,10,15,20,25,31],ticktext:['1','5','10','15','20','25','NC'],zeroline:false},
      shapes:[{type:'line',x0:1,y0:1,x1:30,y1:30,line:{color:'rgba(245,247,250,.6)',width:1}}],
      margin:{l:63,r:63,t:4,b:48},annotations:rows.length?[]:[{xref:'paper',yref:'paper',x:.5,y:.5,text:'No matching entries',showarrow:false}]
    }),CONFIG).then(()=>{
      const chart=$('#heatmapChart'); chart.removeAllListeners?.('plotly_click');
      chart.on('plotly_click',(event)=>{
        const p=event.points[0]; const start=Math.round(p.x); const finish=Math.round(p.y);
        const group=rows.filter(row=>Math.min(size,row.g)===start && outcome(row)+1===finish);
        const label=finish===31?'unclassified':`P${finish}${finish===30?'+':''}`;
        $('#heatmapSelection').textContent=group.length
          ? `Grid ${start}${start===30?'+':''} → ${label}: ${group.length} entries · ${group.slice(0,2).map(row=>`${row.dn}, ${row.y} ${row.rn.replace(' Grand Prix',' GP')}`).join(' · ')}`
          : 'No matching entries. NC means no numeric final classification.';
      });
    });
    $('#heatmapSelection').textContent='Each grid column totals 100%. NC = no numeric classification; positions 30+ are pooled.';
  }

  const BAND_ORDER = ["P1", "P2–P5", "P6–P10", "P11–P15", "P16–P20", "P21+"];

  function probabilityRows() {
    const rows = filteredRecords(false);
    return BAND_ORDER.map((band) => {
      const group = rows.filter((row) => bandFor(row.g) === band);
      const successes = group.filter((row) => row.f !== null && (state.metric === "podium" ? row.f <= 3 : row.f === 1)).length;
      return { band, total: group.length, successes, rate: group.length ? successes / group.length : 0 };
    });
  }

  function renderProbability() {
    const rows = probabilityRows();
    const ordered = [...rows].reverse();
    const color = state.metric === "podium" ? COLORS.blue : COLORS.red;
    const maxRate = Math.max(...rows.map((row) => row.rate), .01);
    const trace = {
      type: "bar",
      orientation: "h",
      y: ordered.map((row) => row.band),
      x: ordered.map((row) => row.rate),
      marker: { color: ordered.map((row) => row.band.startsWith("P1") && row.band !== "P1" ? "rgba(255,59,48,.78)" : color) },
      text: ordered.map((row) => row.total ? percent(row.rate, row.rate < .01 ? 2 : 1) : "n.a."),
      textposition: "outside",
      cliponaxis: false,
      customdata: ordered.map((row) => [row.band, row.total, row.successes]),
      hovertemplate: "%{customdata[0]}<br>%{customdata[2]} successes / %{customdata[1]} entries<br>%{x:.2%}<extra></extra>"
    };
    const layout = plotLayout({
      xaxis: { title: state.metric === "podium" ? "Podium rate" : "Win rate", tickformat: ".0%", range: [0, maxRate * 1.18], gridcolor: COLORS.line, zeroline: false },
      yaxis: { automargin: true, tickfont: { size: 12 } },
      margin: { l: 72, r: 48, t: 4, b: 46 },
      bargap: .36
    });
    Plotly.react("probabilityChart", [trace], layout, CONFIG).then(() => {
      const chart = $("#probabilityChart");
      chart.removeAllListeners?.("plotly_click");
      chart.on("plotly_click", (event) => {
        const band = event.points[0].customdata[0];
        state.gridBand = state.gridBand === band ? null : band;
        $("#probabilityTakeaway").textContent = state.gridBand
          ? `${band} now filters the dashboard. Click the bar again to clear.`
          : "Grid-band focus cleared.";
        renderAll();
      });
    });
    const p11 = rows.filter((row) => ["P11–P15", "P16–P20", "P21+"].includes(row.band));
    const total = p11.reduce((sum, row) => sum + row.total, 0);
    const successes = p11.reduce((sum, row) => sum + row.successes, 0);
    if (!state.gridBand) {
      $("#probabilityTakeaway").textContent = total
        ? `${state.metric === "win" ? "Wins" : "Podiums"} · P1: ${rows[0].successes}/${rows[0].total} · P11+: ${successes}/${integer.format(total)} entries`
        : "No entries match the current filters.";
    }
  }

  function eventKey(row) { return `${row.r}:${row.d}`; }

  function defaultEvent() {
    return sourceData().events.find(row=>row.y===2024&&row.d==='max_verstappen'&&row.rn.includes('São Paulo')) || sourceData().events.at(-1) || null;
  }

  function ensureSelectedEvent(events) {
    if (!events.length) {state.selectedEvent=null;return;}
    const key=state.selectedEvent?eventKey(state.selectedEvent):null;
    state.selectedEvent=events.find(row=>eventKey(row)===key) || events.find(row=>row.y===2024&&row.d==='max_verstappen') || events.at(-1);
  }

  function renderTimeline() {
    const events = filteredEvents();
    ensureSelectedEvent(events);
    const finishMeta = [
      { finish: 1, name: "Winner (P1)", color: COLORS.gold },
      { finish: 2, name: "Finished P2", color: COLORS.silver },
      { finish: 3, name: "Finished P3", color: COLORS.bronze }
    ];
    const traces = finishMeta.map((meta) => {
      const group = events.filter((row) => row.f === meta.finish);
      return {
        type: "scatter",
        mode: "markers",
        name: meta.name,
        x: group.map((row) => row.dt),
        y: group.map((row) => row.g),
        marker: {
          size: group.map((row) => state.selectedEvent && eventKey(row) === eventKey(state.selectedEvent) ? 15 : 8),
          color: group.map((row) => state.selectedEvent && eventKey(row) === eventKey(state.selectedEvent) ? COLORS.red : meta.color),
          line: { color: COLORS.surface, width: 1 }
        },
        customdata: group.map((row) => [eventKey(row), row.dn, row.rn, row.g, row.f, row.og, row.ag]),
        hovertemplate: "<b>%{customdata[1]}</b><br>%{x|%d %b %Y} · %{customdata[2]}<br>P%{customdata[3]} → P%{customdata[4]}<br>Official +%{customdata[5]} · Adjusted +%{customdata[6]}<extra></extra>"
      };
    });
    const layout = plotLayout({
      xaxis: { title: "Race date", type: "date", tickformat: "%Y", gridcolor: COLORS.line, zeroline: false },
      yaxis: { title: "Starting grid", range: [34, 9.5], dtick: 5, gridcolor: COLORS.line, zeroline: false },
      legend: { orientation: "h", x: .43, y: 1.1, bgcolor: "rgba(0,0,0,0)", font: { size: 10 }, itemclick: false, itemdoubleclick: false },
      margin: { l: 58, r: 18, t: 12, b: 45 },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 11, y1: 11, line: { color: "rgba(245,247,250,.6)", width: 1 } }]
    });
    Plotly.react("timelineChart", traces, layout, CONFIG).then(() => {
      const chart = $("#timelineChart");
      chart.removeAllListeners?.("plotly_click");
      chart.on("plotly_click", (event) => {
        const key = event.points[0].customdata[0];
        state.selectedEvent = sourceData().events.find((row) => eventKey(row) === key);
        renderLab();
      });
    });
  }

  function updateEventCard() {
    const row=state.selectedEvent;
    $('#eventDriver').textContent=row?row.dn:'No matching event';
    $('#eventRace').textContent=row?`${row.y} ${row.rn}`:'Change filters or reset to the modern sample.';
    $('#eventRoute').firstElementChild.textContent=row?(row.sn?'Pit lane*':`P${row.g}`):'—';
    $('#eventRoute').lastElementChild.textContent=row?`P${row.f}`:'—';
    $('#eventOfficial').textContent=row?`${row.og>=0?'+':''}${row.og}`:'—';
    $('#eventAdjusted').textContent=row?`${row.ag>=0?'+':''}${row.ag}`:'—';
    $('#eventAttrition').textContent=row?`${row.ssr} → ${row.sfr}`:'—';
    $('#eventMethod').textContent=row?(row.sn||'Re-rank the same finish-status cohort at the start and finish. This is not an overtake or causal estimate.'):'No result is being carried over from a previous filter.';
  }

  function comparisonCohort() {
    const selected=state.selectedEvent;
    if(!selected)return {selected:null,rows:[],scope:''};
    const selectedBand=bandFor(selected.g); const selectedFinish=selected.f===1?'win':'podium';
    const rows=sourceData().events.filter(row=>bandFor(row.g)===selectedBand && (selectedFinish==='win'?row.f===1:row.f>1) && eraMatch(row.y));
    return {selected,rows,scope:state.era==='all'?'1950–2024':state.era,selectedBand,selectedFinish};
  }

  function renderComparison() {
    const {selected,rows,scope,selectedBand,selectedFinish}=comparisonCohort();
    if(!selected||!rows.length){
      $('#comparisonContext').textContent='No matching event. There is no historical fallback outside the selected period.';
      Plotly.react('comparisonChart',[],plotLayout({annotations:[{xref:'paper',yref:'paper',x:.5,y:.5,text:'No comparison available',showarrow:false}]}),CONFIG);return;
    }
    const peers=rows.filter(row=>eventKey(row)!==eventKey(selected)).sort((a,b)=>Math.abs(a.g-selected.g)-Math.abs(b.g-selected.g)||Math.abs(a.f-selected.f)-Math.abs(b.f-selected.f)||Math.abs(a.y-selected.y)-Math.abs(b.y-selected.y)).slice(0,4);
    const visible=[selected,...peers];
    $('#comparisonContext').textContent=`${scope} · ${selectedBand} ${selectedFinish==='win'?'wins':'P2–P3 results'} · pool n=${rows.length}; ${visible.length} shown. Peers match grid, finish and season; driver/circuit filters select the focus only.`;
    const traces=[
      {type:'scatter',mode:'markers',name:'Recorded gain',x:visible.map(r=>r.og),y:visible.map((_,i)=>i),marker:{size:visible.map((_,i)=>i?8:13),color:COLORS.blue},customdata:visible.map(r=>r.dn),hovertemplate:'%{customdata}<br>Recorded gain %{x}<extra></extra>'},
      {type:'scatter',mode:'markers',name:'Finish-cohort gain',x:visible.map(r=>r.ag),y:visible.map((_,i)=>i),marker:{size:visible.map((_,i)=>i?8:13),color:COLORS.red,symbol:'diamond'},customdata:visible.map(r=>[r.dn,r.ssr,r.sfr]),hovertemplate:'%{customdata[0]}<br>Cohort rank %{customdata[1]} → %{customdata[2]}<br>Gain %{x}<extra>Not a causal correction</extra>'}
    ];
    const all=visible.flatMap(r=>[r.og,r.ag]);
    Plotly.react('comparisonChart',traces,plotLayout({
      xaxis:{title:'Positions gained',range:[Math.min(0,...all)-1,Math.max(1,...all)+2],dtick:5,gridcolor:COLORS.line,zeroline:true},
      yaxis:{tickvals:visible.map((_,i)=>i),ticktext:visible.map((r,i)=>`${i===0?'SELECTED · ':''}${r.dn.split(' ').at(-1)} ${r.y} · P${r.g}→P${r.f}`),tickfont:{size:12},range:[visible.length-.45,-.65],fixedrange:true},
      legend:{orientation:'h',x:.45,y:1.25,font:{size:12},itemclick:false,itemdoubleclick:false},
      shapes:visible.map((r,i)=>({type:'line',x0:r.ag,x1:r.og,y0:i,y1:i,line:{color:i?COLORS.line:COLORS.text,width:i?2:4}})),
      margin:{l:225,r:22,t:22,b:40}
    }),CONFIG);
  }

  function renderLab() {
    renderTimeline();
    updateEventCard();
    renderComparison();
  }

  function activeReplay() {
    return sourceData().replays.find((replay) => replay.id === state.replayId) || sourceData().replays[0];
  }

  function replayDrivers(replay) {
    const groups = new Map();
    replay.laps.forEach((row) => {
      if (!groups.has(row.d)) groups.set(row.d, []);
      groups.get(row.d).push(row);
    });
    groups.forEach((rows) => rows.sort((a, b) => a.l - b.l));
    return groups;
  }

  function replayTrace(driver, rows, lap, replay) {
    const visible = rows.filter((row) => row.l <= lap);
    const isFeatured = driver === replay.driver_label;
    return {
      type: "scatter",
      mode: "lines",
      name: isFeatured ? replay.driver_name : driver,
      x: visible.map((row) => row.l),
      y: visible.map((row) => row.p),
      line: { color: isFeatured ? COLORS.red : COLORS.muted, width: isFeatured ? 4.5 : 1.1 },
      opacity: isFeatured ? 1 : .28,
      hovertemplate: `${driver}<br>Lap %{x}<br>P%{y}<extra></extra>`,
      showlegend: isFeatured
    };
  }

  function positionAt(groups, driver, lap) {
    const rows = groups.get(driver) || [];
    const current = rows.find((row) => row.l === lap);
    return current ? current.p : null;
  }

  function updateReplayMetadata(replay, maxLap) {
    const shortRace = replay.race_name.replace(" Grand Prix", " GP");
    $("#replayTitle").textContent = `${replay.year} ${shortRace}: P${replay.grid} to P${replay.finish}`;
    $("#replayDriverName").textContent = replay.driver_name;
    $("#replayStart").textContent = `P${replay.grid}`;
    $("#replayFinalRoute").textContent = `P${replay.grid} → P${replay.finish}`;
    $("#replayFinalGains").textContent = `+${replay.official_gain} recorded · +${replay.adjusted_gain} in finish cohort`;
    $("#lapSlider").max = String(maxLap);
    $("#coverageNote").textContent = `Focus driver: ${replay.observed_laps}/${replay.expected_laps} recorded laps · ${replay.coverage_complete ? "continuous" : "incomplete"}. Final classification P${replay.finish}; last observed lap P${replay.last_lap_position}.`;
    const jumps = $(".lap-jumps");
    jumps.replaceChildren(...replay.key_laps.map((lap) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.lap = String(Math.min(lap, maxLap));
      button.textContent = `Lap ${Math.min(lap, maxLap)}`;
      button.addEventListener("click", () => {
        stopReplay();
        state.replayLap = Number(button.dataset.lap);
        renderReplay();
      });
      return button;
    }));
  }

  function renderReplay() {
    const replay = activeReplay();
    const lapGroups = replayDrivers(replay);
    const maxReplayLap = Math.max(...replay.laps.map((row) => row.l));
    state.replayLap = Math.min(state.replayLap, maxReplayLap);
    updateReplayMetadata(replay, maxReplayLap);
    const traces = [];
    [...lapGroups.entries()]
      .filter(([driver]) => driver !== replay.driver_label)
      .forEach(([driver, rows]) => traces.push(replayTrace(driver, rows, state.replayLap, replay)));
    if (lapGroups.has(replay.driver_label)) {
      traces.push(replayTrace(replay.driver_label, lapGroups.get(replay.driver_label), state.replayLap, replay));
    }
    const snapshot = [...lapGroups.entries()].map(([driver, rows]) => {
      const current = rows.find((row) => row.l === state.replayLap);
      return current ? { driver, row: current } : null;
    }).filter(Boolean);
    traces.push({
      type: "scatter",
      mode: "markers",
      x: snapshot.map(() => state.replayLap),
      y: snapshot.map((item) => item.row.p),
      marker: {
        color: snapshot.map((item) => item.driver === replay.driver_label ? COLORS.red : COLORS.muted),
        size: snapshot.map((item) => item.driver === replay.driver_label ? 11 : 4),
        line: { color: COLORS.surface, width: 1 }
      },
      customdata: snapshot.map((item) => item.driver),
      hovertemplate: "%{customdata}<br>Lap %{x}<br>P%{y}<extra></extra>",
      showlegend: false
    });
    const layout = plotLayout({
      xaxis: { title: "Completed driver lap · 0 = recorded grid", range: [-1, maxReplayLap], dtick: 10, gridcolor: COLORS.line, zeroline: false },
      yaxis: { title: "Race position", range: [Math.max(20,...replay.laps.map(row=>row.p))+.7, .3], dtick: 2, gridcolor: COLORS.line, zeroline: false },
      legend: { orientation: "h", x: .02, y: 1.06, bgcolor: "rgba(0,0,0,0)" },
      margin: { l: 58, r: 20, t: 10, b: 45 }
    });
    Plotly.react("replayChart", traces, layout, CONFIG);
    const position = positionAt(lapGroups, replay.driver_label, state.replayLap);
    const currentRows = replay.laps.filter((row) => row.l === state.replayLap);
    $("#currentPosition").textContent = position ? `P${position}` : "–";
    const liveGain = position ? replay.grid - position : null;
    $("#liveGain").textContent = liveGain === null ? "–" : `${liveGain >= 0 ? "+" : ""}${liveGain}`;
    $("#liveField").textContent = String(new Set(currentRows.map((row) => row.d)).size);
    $("#liveLap").textContent = `Lap ${state.replayLap}`;
    $("#sliderValue").textContent = `Lap ${state.replayLap}`;
    $("#lapSlider").value = String(state.replayLap);
  }

  function stopReplay() {
    if (state.replayTimer) window.clearInterval(state.replayTimer);
    state.replayTimer = null;
  }

  function playReplay() {
    stopReplay();
    const maxReplayLap = Math.max(...activeReplay().laps.map((row) => row.l));
    if (state.replayLap >= maxReplayLap) state.replayLap = 0;
    state.replayTimer = window.setInterval(() => {
      state.replayLap += 1;
      renderReplay();
      if (state.replayLap >= maxReplayLap) stopReplay();
    }, Number($("#replaySpeed").value));
  }

  function renderCircuitChart() {
    const rows = DATA.circuits;
    $("#circuitNote").textContent=`Fixed 1990–2024 exploration, ${rows.length} circuits, reviewed grid values; unaffected by global filters. Pearson r = ${DATA.meta.circuit_correlation.toFixed(2)}. An aggregate correlation does not establish why an individual comeback occurred.`;
    const trace = {
      type: "scatter",
      mode: "markers+text",
      x: rows.map((row) => row.nf),
      y: rows.map((row) => row.er),
      text: rows.map((row) => ["hockenheimring", "americas", "bahrain", "albert_park", "interlagos"].includes(row.c) ? row.cn.replace("Autódromo ", "") : ""),
      textposition: "top center",
      textfont: { color: COLORS.text, size: 11 },
      marker: {
        size: rows.map((row) => Math.sqrt(row.n) * 6.5),
        color: COLORS.blue,
        opacity: .62,
        line: { color: COLORS.text, width: 1 }
      },
      customdata: rows.map((row) => [row.c, row.cn, row.country, row.n, row.p90]),
      hovertemplate: "<b>%{customdata[1]}</b><br>%{customdata[2]}<br>Outside finish cohort: %{x:.1%}<br>Races with P11+ podium: %{y:.1%}<br>Races: %{customdata[3]}<br>P90 adjusted gain: %{customdata[4]}<extra></extra>"
    };
    const layout = plotLayout({
      xaxis: { title: "Share outside finish cohort", tickformat: ".0%", gridcolor: COLORS.line, zeroline: false },
      yaxis: { title: "Share of races with a P11+ podium", tickformat: ".0%", gridcolor: COLORS.line, zeroline: false },
      margin: { l: 72, r: 35, t: 18, b: 58 }
    });
    Plotly.react("circuitChart", [trace], layout, CONFIG).then(() => {
      const chart = $("#circuitChart");
      chart.removeAllListeners?.("plotly_click");
      chart.on("plotly_click", (event) => {
        state.circuit = event.points[0].customdata[0];
        $("#circuitFilter").value = state.circuit;
        $("#circuitDialog").close();
        state.activeTab = "overview";
        switchTab("overview");
      });
    });
  }

  function renderOverview() {
    renderKpis();
    renderHeatmap();
    renderProbability();
  }

  function renderAll() {
    renderKpis();
    if (state.activeTab === "overview") renderOverview();
    if (state.activeTab === "lab") renderLab();
    if (state.activeTab === "replay") renderReplay();
  }

  function switchTab(tab) {
    state.activeTab = tab;
    $$(".tab-button").forEach((button) => {
      const active = button.dataset.tab === tab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    $$(".tab-panel").forEach((panel) => {
      const active = panel.dataset.panel === tab;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    });
    if (tab !== "replay") stopReplay();
    window.requestAnimationFrame(renderAll);
  }

  function populateReplayCases() {
    const select = $("#replayCase");
    sourceData().replays.forEach((replay) => {
      const shortRace = replay.race_name.replace(" Grand Prix", " GP");
      select.add(new Option(`${replay.driver_name} · ${replay.year} ${shortRace}`, replay.id));
    });
    select.value = state.replayId;
  }

  function bindEvents() {
    $("#dataVersion").addEventListener("change", event=>{stopReplay();state.version=event.target.value;state.selectedEvent=null;renderAll();});
    $("#recordMode").addEventListener("click",()=>{document.body.classList.toggle("recording");$("#recordMode").textContent=document.body.classList.contains("recording")?"Exit recording view":"Recording view";renderAll();});
    $("#methodHelp").addEventListener("click",()=>$("#methodDialog").showModal());
    $("#closeMethod").addEventListener("click",()=>$("#methodDialog").close());
    $("#replaySpeed").addEventListener("change",stopReplay);
    $("#replayStartButton").addEventListener("click",()=>{stopReplay();state.replayLap=0;renderReplay();});
    $("#eraFilter").addEventListener("change", (event) => { state.era = event.target.value; state.gridBand = null; renderAll(); });
    $("#driverFilter").addEventListener("change", (event) => { state.driver = event.target.value; state.gridBand = null; renderAll(); });
    $("#circuitFilter").addEventListener("change", (event) => { state.circuit = event.target.value; state.gridBand = null; renderAll(); });
    $("#resetFilters").addEventListener("click", () => {
      Object.assign(state, { era: "2010-2024", driver: "all", circuit: "all", gridBand: null, winsOnly: false });
      $("#eraFilter").value = "2010-2024";
      $("#driverFilter").value = "all";
      $("#circuitFilter").value = "all";
      $("#winsOnly").checked = false;
      renderAll();
    });
    $$(".tab-button").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
    $$(".metric-button").forEach((button) => button.addEventListener("click", () => {
      state.metric = button.dataset.metric;
      $$(".metric-button").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
      renderProbability();
    }));
    $("#winsOnly").addEventListener("change", (event) => { state.winsOnly = event.target.checked; renderLab(); });
    $$(".quick-cases button").forEach((button) => button.addEventListener("click", () => {
      const cases = {
        watson: () => defaultEvent(),
        raikkonen: () => sourceData().events.find((row) => row.y === 2005 && row.d === "raikkonen" && row.rn.includes("Japanese")),
        monteiro: () => sourceData().events.find((row) => row.y === 2005 && row.d === "monteiro" && row.rn.includes("United States")),
        verstappen: () => sourceData().events.find((row) => row.y === 2024 && row.d === "max_verstappen" && row.rn.includes("São Paulo"))
      };
      const chosen = cases[button.dataset.case]?.() || defaultEvent();
      if(chosen){ state.era=chosen.y>=2010?"2010-2024":"all"; $("#eraFilter").value=state.era; state.driver="all";state.circuit="all";state.gridBand=null;state.winsOnly=false;$("#driverFilter").value="all";$("#circuitFilter").value="all";$("#winsOnly").checked=false;state.selectedEvent=chosen; }
      renderAll();
    }));
    $("#replayCase").addEventListener("change", (event) => {
      stopReplay();
      state.replayId = event.target.value;
      state.replayLap = 0;
      renderReplay();
    });
    $("#playReplay").addEventListener("click", playReplay);
    $("#pauseReplay").addEventListener("click", stopReplay);
    $("#lapSlider").addEventListener("input", (event) => { stopReplay(); state.replayLap = Number(event.target.value); renderReplay(); });
    $("#openCircuitDialog").addEventListener("click", () => { $("#circuitDialog").showModal(); window.requestAnimationFrame(renderCircuitChart); });
    $("#closeCircuitDialog").addEventListener("click", () => $("#circuitDialog").close());
    $("#circuitDialog").addEventListener("click", (event) => { if (event.target === $("#circuitDialog")) $("#circuitDialog").close(); });
  }

  function init() {
    if (!DATA || !window.Plotly) {
      $("#sampleLabel").textContent = "Dashboard assets failed to load";
      return;
    }
    state.selectedEvent = defaultEvent();
    state.replayId = sourceData().replays[0]?.id || null;
    populateFilters();
    populateReplayCases();
    bindEvents();
    renderOverview();
  }

  init();
})();
