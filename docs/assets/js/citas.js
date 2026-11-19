// Vincula citas (a.cita) con la lista de referencias APA y avisa de huecos.
// El enunciado penaliza referencias inexistentes: aquí se detectan citas sin
// referencia y referencias que nunca se citan.
(() => {
  const refs = new Map([...document.querySelectorAll(".referencias li[id]")].map(li => [li.id, li]));
  const citadas = new Set();

  document.querySelectorAll("a.cita").forEach((a, i) => {
    const id = a.hash.slice(1);
    const ref = refs.get(id);
    if (!ref) {
      a.classList.add("cita-rota"); a.title = "⚠ referencia no encontrada";
      console.warn(`[citas] cita sin referencia: #${id}`); return;
    }
    a.title = ref.textContent.replace(/\s*↩\s*$/, "").trim();
    if (!citadas.has(id)) {
      citadas.add(id);
      a.id = `cita-${id}`;
      const volver = document.createElement("a");
      volver.className = "volver"; volver.href = `#${a.id}`; volver.textContent = "↩";
      volver.setAttribute("aria-label", "Volver a la primera cita en el texto");
      ref.append(volver);
    }
  });

  refs.forEach((_, id) => { if (!citadas.has(id)) console.warn(`[citas] referencia sin citar: #${id}`); });
})();
