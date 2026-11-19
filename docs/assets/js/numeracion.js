// Numera figuras, tablas y ecuaciones por orden de aparición y resuelve las
// referencias cruzadas <a class="xref" href="#id"></a>.
(() => {
  const grupos = [
    { sel: ".reporte figure.lamina", nombre: "Figura", destino: ".num" },
    { sel: ".reporte .tabla-paper",  nombre: "Tabla",  destino: ".num" },
    { sel: ".reporte .eq",           nombre: "Ecuación", destino: ".eq-num", soloNumero: true },
  ];
  const etiquetas = new Map();

  for (const g of grupos) {
    document.querySelectorAll(g.sel).forEach((el, i) => {
      const n = i + 1;
      const nodo = el.querySelector(g.destino);
      if (nodo) nodo.textContent = g.soloNumero ? String(n) : `${g.nombre} ${n}`;
      if (el.id) etiquetas.set(el.id, `${g.nombre} ${n}`);
    });
  }

  document.querySelectorAll("a.xref").forEach(a => {
    const id = a.hash.slice(1);
    const txt = etiquetas.get(id);
    if (txt) a.textContent = txt;
    else { a.textContent = "⚠ referencia rota"; console.warn(`[numeracion] xref sin destino: #${id}`); }
  });
})();
