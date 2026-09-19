"use strict";
function initializeSite() {
  const menu = document.querySelector(".menu-toggle");
  const nav = document.querySelector("#site-nav");
  if (menu && nav) {
    menu.addEventListener("click", () => {
      const open = menu.getAttribute("aria-expanded") !== "true";
      menu.setAttribute("aria-expanded", String(open));
      nav.classList.toggle("is-open", open);
    });
  }
  const search = document.querySelector("#note-search");
  const select = document.querySelector("#tag-select");
  const cards = Array.from(document.querySelectorAll("#notes-grid .searchable"));
  const count = document.querySelector("#result-count");
  const empty = document.querySelector("#empty-results");
  const tabs = Array.from(document.querySelectorAll(".filter-button"));
  let kind = "all";
  function filterNotes() {
    const query = (search?.value || "").trim().toLocaleLowerCase();
    const tag = select?.value || "";
    let visible = 0;
    cards.forEach(card => {
      const matches = (kind === "all" || card.dataset.kind === kind)
        && (!query || card.dataset.search.includes(query))
        && (!tag || card.dataset.tags.split("|").includes(tag));
      card.hidden = !matches;
      if (matches) visible += 1;
    });
    if (count) count.textContent = `${visible} 篇记录（示例有单独标识）`;
    if (empty) empty.hidden = visible !== 0;
  }
  search?.addEventListener("input", filterNotes);
  select?.addEventListener("change", filterNotes);
  tabs.forEach(tab => tab.addEventListener("click", () => {
    kind = tab.dataset.filter;
    tabs.forEach(item => {
      item.classList.toggle("selected", item === tab);
      item.setAttribute("aria-pressed", String(item === tab));
    });
    filterNotes();
  }));
  if (search) filterNotes();
  document.querySelectorAll(".prose pre").forEach(pre => {
    if (pre.querySelector(".copy-code")) return;
    const button = document.createElement("button");
    button.className = "copy-code";
    button.type = "button";
    button.textContent = "复制代码";
    button.addEventListener("click", async () => {
      const text = pre.querySelector("code")?.textContent || pre.textContent;
      try {
        if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
        await navigator.clipboard.writeText(text);
        button.textContent = "已复制";
      } catch {
        const range = document.createRange();
        const node = pre.querySelector("code");
        if (node) {
          range.selectNodeContents(node);
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
        }
        button.textContent = "已选中，请手动复制";
      }
      setTimeout(() => { button.textContent = "复制代码"; }, 2200);
    });
    pre.appendChild(button);
  });
}
document.addEventListener("keydown", event => {
  const tag = document.activeElement?.tagName;
  if (event.key === "/" && !["INPUT", "TEXTAREA", "SELECT"].includes(tag)
      && !document.activeElement?.isContentEditable) {
    const input = document.querySelector("#note-search");
    if (input) { event.preventDefault(); input.focus(); }
  }
});
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeSite);
} else {
  initializeSite();
}
