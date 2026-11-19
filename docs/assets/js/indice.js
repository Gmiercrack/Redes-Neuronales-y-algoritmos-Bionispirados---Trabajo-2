// Construye el índice desde los h2/h3 de .reporte y mantiene el elemento activo
// (scrollspy). Usa scroll-target-group nativo si existe; si no, IntersectionObserver.
(() => {
  const reporte = document.querySelector(".reporte");
  const contenedor = document.querySelector("#indice");
  if (!reporte || !contenedor) return;

  const slug = t => t.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
                     .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

  const raiz = document.createElement("ol");
  let nSec = 0, nSub = 0, itemActual = null, subActual = null, parteActual = null;
  const nombres = { pd: "Parte 1 · PD", lgd: "Parte 2 · LGD", el: "Pérdida esperada" };

  reporte.querySelectorAll("section > h2, section > h3").forEach(h => {
    if (!h.id) h.id = slug(h.textContent);
    const esH2 = h.tagName === "H2";
    const sec = h.closest("section[data-parte]");
    const parte = sec?.dataset.parte ?? null;

    if (esH2) {
      if (!h.classList.contains("sin-num")) { nSec++; nSub = 0; h.dataset.num = nSec; }
      if (parte && parte !== parteActual) {
        const g = document.createElement("li"); g.className = "grupo"; g.textContent = nombres[parte] ?? parte;
        g.style.setProperty("--color-parte", `var(--${parte})`);
        raiz.append(g);
      }
      parteActual = parte;
      itemActual = document.createElement("li");
      const a = document.createElement("a"); a.href = `#${h.id}`;
      a.textContent = (h.dataset.num ? h.dataset.num + ". " : "") + h.textContent;
      itemActual.append(a); raiz.append(itemActual); subActual = null;
    } else if (itemActual) {
      if (!h.classList.contains("sin-num")) { nSub++; h.dataset.num = `${nSec}.${nSub}`; }
      if (!subActual) { subActual = document.createElement("ol"); itemActual.append(subActual); }
      const li = document.createElement("li"); const a = document.createElement("a");
      a.href = `#${h.id}`; a.textContent = (h.dataset.num ? h.dataset.num + " " : "") + h.textContent;
      li.append(a); subActual.append(li);
    }
  });
  contenedor.append(raiz);

  // Respaldo para navegadores sin scroll-target-group (Firefox, Safari)
  const enlaces = [...document.querySelectorAll("#indice a")];
  const sincronizarAria = actual => enlaces.forEach(a => {
    a.setAttribute("aria-current", a === actual ? "true" : "false");
  });

  if (CSS.supports("scroll-target-group: auto")) {
    const sync = () => sincronizarAria(document.querySelector("#indice a:target-current"));
    sync();
    document.addEventListener("scrollend", sync);
  } else {
    const titulos = [...reporte.querySelectorAll("h2[id], h3[id]")];
    const io = new IntersectionObserver(entradas => {
      for (const e of entradas) if (e.isIntersecting) {
        const a = enlaces.find(x => x.hash === `#${e.target.id}`);
        enlaces.forEach(x => x.classList.toggle("activo", x === a));
        sincronizarAria(a);
      }
    }, { rootMargin: "-20% 0px -70% 0px", threshold: 0 });
    titulos.forEach(t => io.observe(t));
  }
})();
