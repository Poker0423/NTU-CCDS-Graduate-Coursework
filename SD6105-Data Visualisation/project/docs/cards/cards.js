(() => {
  const config = window.VIDEO_CARD_CONFIG || {};
  const setText = (selector, value) => {
    const node = document.querySelector(selector);
    if (node && value) node.textContent = value;
  };
  setText("[data-course]", config.course);
  setText("[data-presenter-a]", config.presenterA);
  setText("[data-presenter-b]", config.presenterB);
  setText("[data-source]", config.source);
})();
