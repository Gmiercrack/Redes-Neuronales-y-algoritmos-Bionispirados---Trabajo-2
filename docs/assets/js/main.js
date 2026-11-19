// Botón Copiar, panel de índice móvil y ecuaciones con KaTeX.
(() => {
  // Copiar código
  document.querySelectorAll("pre").forEach(pre => {
    const b = document.createElement("button");
    b.className = "copiar"; b.type = "button"; b.textContent = "Copiar";
    b.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(pre.innerText.replace(/Copiar$/, "").trim()); b.textContent = "Copiado"; }
      catch { b.textContent = "No se pudo"; }
      setTimeout(() => (b.textContent = "Copiar"), 1500);
    });
    pre.append(b);
  });

  // Panel de índice en móvil (<dialog> nativo)
  const dlg = document.querySelector("#panel-indice");
  const abrir = document.querySelector("#abrir-indice");
  const indice = document.querySelector("#indice");
  if (dlg && abrir && indice) {
    const copia = indice.cloneNode(true); copia.removeAttribute("id");
    copia.querySelectorAll("a").forEach(a => a.addEventListener("click", () => dlg.close()));
    dlg.querySelector(".contenido").append(copia);
    abrir.addEventListener("click", () => dlg.showModal());
    dlg.querySelector(".cerrar").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });
  }

  // KaTeX (CDN); si no hay red, las fórmulas quedan como TeX legible
  const renderizar = () => window.renderMathInElement?.(document.querySelector(".reporte"), {
    delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
    throwOnError: false,
  });
  if (window.renderMathInElement) renderizar();
  else window.addEventListener("load", renderizar);
})();
