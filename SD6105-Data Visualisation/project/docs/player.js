(() => {
  const scenes = [
    { id: "title", label: "Title", src: "cards/title.html" },
    { id: "dashboard", label: "Dashboard", src: "dashboard/index.html" },
    { id: "conclusion", label: "Conclusion", src: "cards/conclusion.html" }
  ];
  const frame = document.querySelector("#sceneFrame");
  const status = document.querySelector("#sceneStatus");
  const previous = document.querySelector("#previousScene");
  const next = document.querySelector("#nextScene");
  const hide = document.querySelector("#hideControls");
  const reveal = document.querySelector(".reveal-zone");
  const tabs = [...document.querySelectorAll("[data-scene]")];
  let index = 0;

  const showScene = (requested, updateHash = true) => {
    const nextIndex = typeof requested === "number"
      ? Math.max(0, Math.min(scenes.length - 1, requested))
      : Math.max(0, scenes.findIndex(scene => scene.id === requested));
    index = nextIndex;
    const scene = scenes[index];
    if (!frame.src.endsWith(scene.src)) frame.src = scene.src;
    frame.title = `${scene.label} scene`;
    status.textContent = `Scene ${index + 1} of ${scenes.length}`;
    previous.disabled = index === 0;
    next.disabled = index === scenes.length - 1;
    tabs.forEach(button => {
      const active = button.dataset.scene === scene.id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-current", active ? "page" : "false");
    });
    if (updateHash) history.replaceState(null, "", `#${scene.id}`);
  };

  const toggleControls = (hidden) => {
    document.body.classList.toggle("controls-hidden", hidden);
    hide.textContent = hidden ? "Show controls (H)" : "Hide controls (H)";
  };

  tabs.forEach(button => button.addEventListener("click", () => showScene(button.dataset.scene)));
  previous.addEventListener("click", () => showScene(index - 1));
  next.addEventListener("click", () => showScene(index + 1));
  hide.addEventListener("click", () => toggleControls(!document.body.classList.contains("controls-hidden")));
  reveal.addEventListener("pointerenter", () => toggleControls(false));

  window.addEventListener("keydown", event => {
    if (event.key.toLowerCase() === "h") toggleControls(!document.body.classList.contains("controls-hidden"));
    if (event.key === "ArrowLeft") showScene(index - 1);
    if (event.key === "ArrowRight") showScene(index + 1);
    if (/^[1-3]$/.test(event.key)) showScene(Number(event.key) - 1);
  });

  const initial = location.hash.slice(1);
  showScene(scenes.some(scene => scene.id === initial) ? initial : "title", false);
})();
