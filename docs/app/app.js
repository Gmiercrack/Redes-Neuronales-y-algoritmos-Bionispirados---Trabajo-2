// ============================================================================
// Polyfills condicionales con detección de características (solo navegadores antiguos)
// ============================================================================
if (typeof HTMLButtonElement !== 'undefined' && !('commandForElement' in HTMLButtonElement.prototype)) {
  import('https://esm.run/invokers-polyfill').catch(() => {});
}
if (typeof HTMLElement !== 'undefined' && !Object.hasOwn(HTMLElement.prototype, 'popover')) {
  import('https://unpkg.com/@oddbird/popover-polyfill@latest/dist/popover.min.js').catch(() => {});
}

// ============================================================================
// Motor matemático puro (idéntico al preprocesador, MLP y Scorecard de Python)
// ============================================================================
export function calcularCuotaFrancesa(monto, tasaAnualPct, plazoMeses) {
  const r = (tasaAnualPct / 100) / 12;
  if (!(monto > 0) || !(plazoMeses > 0)) return 0;
  if (r <= 0) return Math.round((monto / plazoMeses) * 100) / 100;
  const factor = (1 + r) ** plazoMeses;
  return Math.round(monto * ((r * factor) / (factor - 1)) * 100) / 100;
}

export function preprocesarPerfil(perfil, prep) {
  const { columnas, medianas, indicadores_indices, media, escala } = prep.numericas;
  const vecNum = [];
  const faltantes = [];

  for (let i = 0; i < columnas.length; i += 1) {
    const col = columnas[i];
    const crudo = perfil[col];
    const esFaltante = crudo === null || crudo === undefined || crudo === '' || Number.isNaN(Number(crudo));
    faltantes.push(esFaltante);
    vecNum.push(esFaltante ? medianas[i] : Number(crudo));
  }

  for (const idx of indicadores_indices) {
    vecNum.push(faltantes[idx] ? 1.0 : 0.0);
  }

  const salida = [];
  for (let k = 0; k < vecNum.length; k += 1) {
    salida.push((vecNum[k] - media[k]) / escala[k]);
  }

  for (const cat of prep.categoricas) {
    const crudo = perfil[cat.nombre];
    const valor = crudo === null || crudo === undefined || crudo === '' ? cat.moda : String(crudo);
    let enFrecuentes = false;
    for (const freq of cat.frecuentes) {
      const activo = valor === freq ? 1.0 : 0.0;
      if (activo === 1.0) enFrecuentes = true;
      salida.push(activo);
    }
    if (cat.tiene_infrecuente) {
      salida.push(enFrecuentes ? 0.0 : 1.0);
    }
  }

  return salida;
}

function activar(valor, tipo) {
  if (tipo === 'relu') return valor > 0 ? valor : 0;
  if (tipo === 'tanh') return Math.tanh(valor);
  if (tipo === 'elu') return valor >= 0 ? valor : Math.exp(valor) - 1;
  throw new Error(`Activación no soportada: ${tipo}`);
}

export function predecirMLP(x, modeloMlp) {
  let h = x;
  const { capas, activacion } = modeloMlp;
  for (let idx = 0; idx < capas.length; idx += 1) {
    const { W, b } = capas[idx];
    const siguiente = new Array(W.length);
    const esUltima = idx === capas.length - 1;
    for (let i = 0; i < W.length; i += 1) {
      const fila = W[i];
      let suma = b[i];
      for (let j = 0; j < fila.length; j += 1) {
        suma += fila[j] * h[j];
      }
      siguiente[i] = esUltima ? suma : activar(suma, activacion);
    }
    h = siguiente;
  }
  const z = Math.max(-60, Math.min(60, h[0]));
  return 1 / (1 + Math.exp(-z));
}

export function predecirLogistica(x, lr) {
  let z = lr.intercepto;
  for (let i = 0; i < lr.coeficientes.length; i += 1) {
    z += lr.coeficientes[i] * x[i];
  }
  const zClip = Math.max(-60, Math.min(60, z));
  return 1 / (1 + Math.exp(-zClip));
}

export function probAPuntaje(p, scorecard) {
  const pClip = Math.max(1e-15, Math.min(1 - 1e-15, p));
  return scorecard.A + scorecard.B * Math.log((1 - pClip) / pClip);
}

export function calcularPercentil(puntaje, percentiles101) {
  if (!Array.isArray(percentiles101) || percentiles101.length < 2) return 50;
  if (puntaje <= percentiles101[0]) return 0;
  const ultimo = percentiles101.length - 1;
  if (puntaje >= percentiles101[ultimo]) return 100;
  for (let q = 1; q < percentiles101.length; q += 1) {
    const izq = percentiles101[q - 1];
    const der = percentiles101[q];
    if (puntaje <= der) {
      const span = der - izq;
      const frac = span > 0 ? (puntaje - izq) / span : 0;
      return q - 1 + frac;
    }
  }
  return 50;
}

export function clasificarBanda(puntaje, bandas) {
  for (let i = 0; i < bandas.length; i += 1) {
    if (puntaje <= bandas[i].puntaje_max || i === bandas.length - 1) {
      return bandas[i];
    }
  }
  return bandas[0];
}

export function calcularContribucionesVariables(x, modelo) {
  const { regresion_logistica: lr, mapa_variables: mapa } = modelo;
  return mapa.map((grupo) => {
    let deltaPuntos = 0;
    for (const idx of grupo.indices) {
      deltaPuntos += lr.pesos_puntos[idx] * (x[idx] - lr.media_cartera_x[idx]);
    }
    return {
      variable: grupo.variable,
      etiqueta: grupo.etiqueta,
      deltaPuntos,
    };
  });
}

export function evaluarPerfilCompleto(perfil, modelo) {
  const x = preprocesarPerfil(perfil, modelo.preprocesador);
  const pdMlp = predecirMLP(x, modelo.mlp_pd);
  const puntajeMlp = probAPuntaje(pdMlp, modelo.scorecard);
  const pdLr = predecirLogistica(x, modelo.regresion_logistica);
  const puntajeLr = probAPuntaje(pdLr, modelo.scorecard);
  const percentil = calcularPercentil(puntajeMlp, modelo.scorecard.percentiles_101);
  const banda = clasificarBanda(puntajeMlp, modelo.scorecard.bandas);
  const contribuciones = calcularContribucionesVariables(x, modelo);

  let parte2 = null;
  if (modelo.lgd_parte2) {
    const lgd = predecirMLP(x, modelo.lgd_parte2);
    const fracs = modelo.lgd_parte2.ead_fraccion_por_plazo;
    const plazoClave = String(Math.round(Number(perfil.plazo_meses)));
    const factorEad = fracs[plazoClave] ?? fracs.todos ?? 0.65;
    const monto = Number(perfil.funded_amnt ?? perfil.loan_amnt);
    const ead = monto * factorEad;
    const el = pdMlp * lgd * ead;
    parte2 = { lgd, factorEad, ead, el, elPctMonto: monto > 0 ? el / monto : 0 };
  }

  return { x, pdMlp, puntajeMlp, pdLr, puntajeLr, percentil, banda, contribuciones, parte2 };
}

// ============================================================================
// Formateadores en español y validación post-interacción
// ============================================================================
const fmtNum = new Intl.NumberFormat('es-CO', { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const fmtPct = new Intl.NumberFormat('es-CO', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
const fmtUsd = new Intl.NumberFormat('es-CO', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

const ETIQUETAS_SLIDERS = {
  int_rate: (v) => `${fmtPct.format(v)} %`,
  annual_inc: (v) => fmtUsd.format(v),
  revol_util: (v) => `${fmtNum.format(v)} %`,
  dti: (v) => `${fmtNum.format(v)} %`,
};

function configurarValidacionPostInteraccion(form, anunciador) {
  const dirtyState = new WeakMap();
  const flagged = new WeakSet();
  const announced = new WeakSet();
  let announceTimer;
  let announcedField;

  const linkError = (el, shown) => {
    const errorId = el.dataset.error;
    if (!errorId) return;
    const ids = (el.getAttribute('aria-describedby') || '')
      .split(/\s+/)
      .filter((id) => id && id !== errorId);
    if (shown) ids.push(errorId);
    if (ids.length) el.setAttribute('aria-describedby', ids.join(' '));
    else el.removeAttribute('aria-describedby');
  };

  const mensajeErrorEspanol = (el) => {
    if (el.validity.valueMissing) return 'Este campo es obligatorio.';
    if (el.validity.rangeUnderflow) return `Ingresa un valor mayor o igual a ${el.min}.`;
    if (el.validity.rangeOverflow) return `Ingresa un valor menor o igual a ${el.max}.`;
    if (el.validity.badInput) return 'Ingresa un número válido.';
    return 'Revisa el valor ingresado.';
  };

  const spokenText = (node) => {
    const copy = node.cloneNode(true);
    copy.querySelectorAll('[aria-hidden="true"]').forEach((n) => n.remove());
    copy.querySelectorAll('[role="img"][aria-label]').forEach((n) => {
      n.replaceWith(`${n.getAttribute('aria-label')}: `);
    });
    return copy.textContent.replace(/\s+/g, ' ').trim();
  };

  const announceError = (el) => {
    const error = el.dataset.error ? document.getElementById(el.dataset.error) : null;
    if (!error || !anunciador) return;
    const label = el.labels?.[0] ? `${spokenText(el.labels[0])}: ` : '';
    clearTimeout(announceTimer);
    anunciador.textContent = '';
    announcedField = el;
    announceTimer = setTimeout(() => {
      anunciador.textContent = label + spokenText(error);
    }, 500);
  };

  const syncAria = (el, announce = false) => {
    if (!el.matches?.('input, select')) return;
    if (el.type === 'radio' || el.type === 'checkbox') return;
    const soportaUserInvalid = CSS.supports('selector(:user-invalid)');
    const state = dirtyState.get(el);
    const invalid = soportaUserInvalid
      ? el.matches(':user-invalid')
      : Boolean(state?.hasBlurred && !el.validity.valid);

    if (!soportaUserInvalid && state?.hasBlurred) {
      el.classList.toggle('user-invalid-fallback', !el.validity.valid);
    }

    if (invalid) {
      const errEl = el.dataset.error ? document.getElementById(el.dataset.error) : null;
      const errTxt = errEl?.querySelector('.err-texto');
      if (errTxt) errTxt.textContent = mensajeErrorEspanol(el);
      if (announce && !announced.has(el)) {
        announceError(el);
        announced.add(el);
      }
      el.setAttribute('aria-invalid', 'true');
      linkError(el, true);
      flagged.add(el);
    } else if (flagged.has(el) || el.getAttribute('aria-invalid') !== 'true') {
      el.classList.remove('user-invalid-fallback');
      el.setAttribute('aria-invalid', 'false');
      linkError(el, false);
      flagged.delete(el);
      announced.delete(el);
      if (announcedField === el) {
        clearTimeout(announceTimer);
        anunciador.textContent = '';
      }
    }
  };

  form.querySelectorAll('input, select').forEach((el) => {
    if (el.type !== 'radio' && el.type !== 'checkbox') {
      el.setAttribute('aria-invalid', 'false');
    }
  });

  form.addEventListener('blur', (e) => {
    const el = e.target;
    const st = dirtyState.get(el) || { hasInteracted: false, hasBlurred: false };
    st.hasBlurred = true;
    dirtyState.set(el, st);
    syncAria(el, true);
  }, true);

  form.addEventListener('input', (e) => {
    const el = e.target;
    const st = dirtyState.get(el) || { hasInteracted: false, hasBlurred: false };
    st.hasInteracted = true;
    dirtyState.set(el, st);
    if (el.hasAttribute('aria-invalid')) syncAria(el, false);
  });

  form.addEventListener('change', (e) => {
    syncAria(e.target, true);
  }, true);

  form.addEventListener('invalid', (e) => {
    const el = e.target;
    dirtyState.set(el, { hasInteracted: true, hasBlurred: true });
    announced.add(el);
    syncAria(el, false);
  }, true);

  return { syncTodos: () => form.querySelectorAll('input, select').forEach((el) => syncAria(el, false)) };
}

// ============================================================================
// Controlador UI
// ============================================================================
async function iniciarAplicacion() {
  if (typeof document === 'undefined') return;

  const anunciador = document.getElementById('anunciador-estado');
  const form = document.getElementById('form-perfil');
  const bannerError = document.getElementById('banner-error');

  let modelo;
  try {
    const resp = await fetch('data/modelo_web.json', { cache: 'no-cache' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    modelo = await resp.json();
  } catch (err) {
    if (bannerError) bannerError.hidden = false;
    console.error('Error cargando modelo_web.json:', err);
    return;
  }

  const validador = configurarValidacionPostInteraccion(form, anunciador);

  // Estado
  let presetActivoId = 'perfil_medio';
  let perfilBase = null;
  let evaluacionBase = null;
  let ultimoEvaluado = null;

  const chkSinAntig = document.getElementById('chk-sin-antiguedad');
  const chkAutoCuota = document.getElementById('chk-auto-cuota');
  const inpAntig = document.getElementById('inp-antiguedad_empleo');
  const inpCuota = document.getElementById('inp-installment');
  const inpFondeo = document.getElementById('inp-funded_amnt');

  const mapaSliders = {
    int_rate: document.getElementById('sld-int_rate'),
    annual_inc: document.getElementById('sld-annual_inc'),
    revol_util: document.getElementById('sld-revol_util'),
    dti: document.getElementById('sld-dti'),
  };

  function leerPerfilDesdeFormulario() {
    const fd = new FormData(form);
    const num = (k, defecto) => {
      const raw = fd.get(k);
      if (raw === null || String(raw).trim() === '') return defecto;
      const v = Number(raw);
      return Number.isFinite(v) ? v : defecto;
    };
    const monto = num('loan_amnt', 14000);
    const tasa = num('int_rate', 13.5);
    const plazo = Number(fd.get('plazo_meses') || 36);
    const cuota = chkAutoCuota.checked
      ? calcularCuotaFrancesa(monto, tasa, plazo)
      : num('installment', 475);
    const fondeo = chkAutoCuota.checked ? monto : num('funded_amnt', monto);

    return {
      loan_amnt: monto,
      funded_amnt: fondeo,
      int_rate: tasa,
      installment: cuota,
      annual_inc: num('annual_inc', 65000),
      dti: num('dti', 17.5),
      delinq_2yrs: num('delinq_2yrs', 0),
      inq_last_6mths: num('inq_last_6mths', 1),
      open_acc: num('open_acc', 10),
      pub_rec: num('pub_rec', 0),
      revol_bal: num('revol_bal', 12000),
      revol_util: num('revol_util', 55),
      total_acc: num('total_acc', 23),
      plazo_meses: plazo,
      antiguedad_empleo: chkSinAntig.checked ? null : num('antiguedad_empleo', 6),
      meses_historial_credito: num('meses_historial_credito', 160),
      grade: String(fd.get('grade') || 'C'),
      home_ownership: String(fd.get('home_ownership') || 'RENT'),
      verification_status: String(fd.get('verification_status') || 'Source Verified'),
      purpose: String(fd.get('purpose') || 'debt_consolidation'),
      initial_list_status: String(fd.get('initial_list_status') || 'f'),
    };
  }

  function sincronizarCamposDependientes() {
    if (chkSinAntig.checked) {
      inpAntig.disabled = true;
      inpAntig.removeAttribute('required');
    } else {
      inpAntig.disabled = false;
      inpAntig.setAttribute('required', '');
      if (inpAntig.value.trim() === '') inpAntig.value = '5';
    }
    if (chkAutoCuota.checked) {
      const monto = Number(document.getElementById('inp-loan_amnt').value) || 0;
      const tasa = Number(document.getElementById('inp-int_rate').value) || 0;
      const plazo = Number(form.querySelector('input[name="plazo_meses"]:checked')?.value || 36);
      inpCuota.value = String(calcularCuotaFrancesa(monto, tasa, plazo));
      inpFondeo.value = String(monto);
      inpCuota.disabled = true;
      inpFondeo.disabled = true;
    } else {
      inpCuota.disabled = false;
      inpFondeo.disabled = false;
    }
  }

  function escribirPerfilEnFormulario(valores) {
    for (const [clave, val] of Object.entries(valores)) {
      if (clave === 'antiguedad_empleo') {
        const falta = val === null || val === undefined;
        chkSinAntig.checked = falta;
        inpAntig.value = falta ? '' : String(val);
        continue;
      }
      if (clave === 'plazo_meses' || clave === 'initial_list_status') {
        const radio = form.querySelector(`input[name="${clave}"][value="${val}"]`);
        if (radio) radio.checked = true;
        continue;
      }
      const el = form.elements.namedItem(clave);
      if (el && 'value' in el) el.value = String(val);
    }
    sincronizarCamposDependientes();
    sincronizarSlidersWhatIf();
    validador.syncTodos();
  }

  function sincronizarSlidersWhatIf() {
    for (const [clave, slider] of Object.entries(mapaSliders)) {
      const el = form.elements.namedItem(clave);
      if (!el || !slider) continue;
      const val = Number(el.value);
      if (Number.isFinite(val)) {
        slider.value = String(Math.min(Number(slider.max), Math.max(Number(slider.min), val)));
        const lblVal = document.getElementById(`val-sld-${clave}`);
        if (lblVal) lblVal.textContent = ETIQUETAS_SLIDERS[clave](val);
      }
    }
    const plazo = Number(form.querySelector('input[name="plazo_meses"]:checked')?.value || 36);
    document.getElementById('btn-whatif-36').setAttribute('aria-pressed', String(plazo === 36));
    document.getElementById('btn-whatif-60').setAttribute('aria-pressed', String(plazo === 60));
  }

  function construirTablaBandas(bandaActivaId) {
    const tbody = document.getElementById('tbody-bandas');
    tbody.textContent = '';
    for (const b of modelo.scorecard.bandas) {
      const tr = document.createElement('tr');
      if (b.banda === bandaActivaId) tr.classList.add('banda-activa');
      const celdas = [
        `D${b.banda}${b.banda === bandaActivaId ? ' (actual)' : ''}`,
        `${fmtNum.format(b.puntaje_min)} – ${fmtNum.format(b.puntaje_max)}`,
        b.creditos.toLocaleString('es-CO'),
        `${fmtPct.format(b.pd_media_predicha * 100)} %`,
        `${fmtPct.format(b.tasa_malos_observada * 100)} %`,
      ];
      celdas.forEach((txt, idx) => {
        const cell = idx === 0 ? document.createElement('th') : document.createElement('td');
        if (idx === 0) cell.scope = 'row';
        cell.textContent = txt;
        tr.appendChild(cell);
      });
      tbody.appendChild(tr);
    }
  }

  function renderizarHistograma(puntajeActual, percentil) {
    const svg = document.getElementById('svg-poblacion');
    const hist = modelo.scorecard.histograma;
    const bordes = hist.bordes;
    const minX = bordes[0];
    const maxX = bordes[bordes.length - 1];
    const W = 640;
    const H = 240;
    const m = { l: 42, r: 18, t: 28, b: 36 };
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;

    const maxDens = Math.max(...hist.densidad_buenos, ...hist.densidad_malos, 1e-6) * 1.12;
    const mapX = (s) => m.l + ((Math.max(minX, Math.min(maxX, s)) - minX) / (maxX - minX)) * pw;
    const mapY = (d) => m.t + ph - (d / maxDens) * ph;

    const crearPoligono = (dens) => {
      let pts = `${mapX(bordes[0]).toFixed(1)},${(m.t + ph).toFixed(1)}`;
      for (let i = 0; i < dens.length; i += 1) {
        const x0 = mapX(bordes[i]).toFixed(1);
        const x1 = mapX(bordes[i + 1]).toFixed(1);
        const y = mapY(dens[i]).toFixed(1);
        pts += ` ${x0},${y} ${x1},${y}`;
      }
      pts += ` ${mapX(bordes[bordes.length - 1]).toFixed(1)},${(m.t + ph).toFixed(1)}`;
      return pts;
    };

    const pasosTicks = [480, 500, 520, 540, 560, 580, 600, 620, 640].filter((t) => t >= minX && t <= maxX);
    const ticksSvg = pasosTicks
      .map((t) => {
        const px = mapX(t).toFixed(1);
        return `<line x1="${px}" y1="${m.t + ph}" x2="${px}" y2="${m.t + ph + 5}" stroke="#65635E" stroke-width="1"/>
                <text x="${px}" y="${m.t + ph + 19}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="#65635E">${t}</text>`;
      })
      .join('');

    const xMedB = mapX(hist.mediana_buenos).toFixed(1);
    const xMedM = mapX(hist.mediana_malos).toFixed(1);
    const xAct = mapX(puntajeActual).toFixed(1);

    let marcadorBaseSvg = '';
    if (evaluacionBase) {
      const xB = mapX(evaluacionBase.puntajeMlp).toFixed(1);
      marcadorBaseSvg = `
        <line x1="${xB}" y1="${m.t}" x2="${xB}" y2="${m.t + ph}" stroke="#2F5D50" stroke-width="2" stroke-dasharray="4 3"/>
        <text x="${xB}" y="${m.t + 12}" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10" font-weight="700" fill="#2F5D50">Base (${fmtNum.format(evaluacionBase.puntajeMlp)})</text>`;
    }

    const spanX = maxX - minX;
    const etiquetaAlineacion = puntajeActual > minX + spanX * 0.78
      ? 'end'
      : puntajeActual < minX + spanX * 0.22
        ? 'start'
        : 'middle';

    svg.innerHTML = `
      <line x1="${m.l}" y1="${m.t + ph}" x2="${W - m.r}" y2="${m.t + ph}" stroke="#D6D2C8" stroke-width="1"/>
      <polygon points="${crearPoligono(hist.densidad_buenos)}" fill="rgba(42,120,214,0.34)" stroke="#2a78d6" stroke-width="1.3"/>
      <polygon points="${crearPoligono(hist.densidad_malos)}" fill="rgba(235,104,52,0.34)" stroke="#eb6834" stroke-width="1.3"/>
      <line x1="${xMedM}" y1="${m.t + 16}" x2="${xMedM}" y2="${m.t + ph}" stroke="#eb6834" stroke-width="1.2" stroke-dasharray="3 3"/>
      <line x1="${xMedB}" y1="${m.t + 16}" x2="${xMedB}" y2="${m.t + ph}" stroke="#2a78d6" stroke-width="1.2" stroke-dasharray="3 3"/>
      ${marcadorBaseSvg}
      <line x1="${xAct}" y1="${m.t - 4}" x2="${xAct}" y2="${m.t + ph}" stroke="#2A2A2A" stroke-width="2.5"/>
      <circle cx="${xAct}" cy="${m.t + 8}" r="4.5" fill="#2A2A2A"/>
      <text x="${xAct}" y="${m.t - 9}" text-anchor="${etiquetaAlineacion}" font-family="IBM Plex Sans, sans-serif" font-size="11.5" font-weight="700" fill="#2A2A2A">
        Tu perfil: ${fmtNum.format(puntajeActual)} (P${Math.round(percentil)})
      </text>
      ${ticksSvg}
      <text x="${(m.l + pw / 2).toFixed(1)}" y="${H - 4}" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="#65635E">
        Puntaje Scorecard (MLP final · mediana malos ${hist.mediana_malos.toFixed(1)} vs. buenos ${hist.mediana_buenos.toFixed(1)})
      </text>
    `;

    svg.setAttribute(
      'aria-label',
      `Distribución de puntajes en prueba (n=${hist.n_prueba}). Mediana de malos: ${hist.mediana_malos.toFixed(1)}; mediana de buenos: ${hist.mediana_buenos.toFixed(1)}. Tu perfil obtiene ${fmtNum.format(puntajeActual)} puntos, ubicándose en el percentil ${fmtNum.format(percentil)}.`
    );

    const caption = document.getElementById('caption-poblacion');
    caption.textContent =
      `Tu puntaje (${fmtNum.format(puntajeActual)}) supera al ${fmtNum.format(percentil)} % de los ${hist.n_prueba.toLocaleString('es-CO')} créditos evaluados en prueba ` +
      `(mediana de buenos: ${hist.mediana_buenos.toFixed(1)}; mediana de malos: ${hist.mediana_malos.toFixed(1)}).`;
  }

  function renderizarPalancasMlp(perfil, puntajeActual) {
    const lista = document.getElementById('lista-palancas-mlp');
    lista.textContent = '';

    const experimentos = [];
    if (perfil.revol_util > 10) {
      const nuevaUtil = Math.max(5, Math.round(perfil.revol_util - 20));
      experimentos.push({
        titulo: `Reducir utilización revolvente de ${fmtNum.format(perfil.revol_util)} % a ${nuevaUtil} %`,
        cambio: { revol_util: nuevaUtil },
      });
    }
    if (perfil.dti > 6) {
      const nuevoDti = Math.max(5, Number((perfil.dti - 5).toFixed(1)));
      experimentos.push({
        titulo: `Reducir relación DTI de ${fmtNum.format(perfil.dti)} % a ${fmtNum.format(nuevoDti)} %`,
        cambio: { dti: nuevoDti },
      });
    }
    if (perfil.plazo_meses === 60) {
      experimentos.push({
        titulo: 'Cambiar el plazo de 60 meses a 36 meses',
        cambio: {
          plazo_meses: 36,
          installment: calcularCuotaFrancesa(perfil.loan_amnt, perfil.int_rate, 36),
        },
      });
    } else {
      experimentos.push({
        titulo: 'Extender el plazo de 36 meses a 60 meses (efecto de riesgo)',
        cambio: {
          plazo_meses: 60,
          installment: calcularCuotaFrancesa(perfil.loan_amnt, perfil.int_rate, 60),
        },
      });
    }
    if (perfil.int_rate > 8) {
      const nuevaTasa = Math.max(6.5, Number((perfil.int_rate - 3).toFixed(1)));
      experimentos.push({
        titulo: `Mejorar tasa de ${fmtPct.format(perfil.int_rate)} % a ${fmtPct.format(nuevaTasa)} %`,
        cambio: {
          int_rate: nuevaTasa,
          installment: calcularCuotaFrancesa(perfil.loan_amnt, nuevaTasa, perfil.plazo_meses),
        },
      });
    }
    if (perfil.antiguedad_empleo === null) {
      experimentos.push({
        titulo: 'Reportar antigüedad laboral de 5 años (quitar dato faltante)',
        cambio: { antiguedad_empleo: 5 },
      });
    }

    const resultados = experimentos
      .map((exp) => {
        const simulado = { ...perfil, ...exp.cambio };
        const pdSim = predecirMLP(preprocesarPerfil(simulado, modelo.preprocesador), modelo.mlp_pd);
        const ptsSim = probAPuntaje(pdSim, modelo.scorecard);
        return { titulo: exp.titulo, deltaPts: ptsSim - puntajeActual, pdSim };
      })
      .sort((a, b) => Math.abs(b.deltaPts) - Math.abs(a.deltaPts));

    for (const r of resultados) {
      const li = document.createElement('li');
      const spanTxt = document.createElement('span');
      spanTxt.textContent = r.titulo;
      const spanVal = document.createElement('span');
      const signo = r.deltaPts >= 0 ? '+' : '';
      spanVal.className = `palanca-delta ${r.deltaPts >= 0 ? 'pos' : 'neg'}`;
      spanVal.textContent = `${signo}${fmtNum.format(r.deltaPts)} pts (PD ${fmtPct.format(r.pdSim * 100)} %)`;
      li.append(spanTxt, spanVal);
      lista.appendChild(li);
    }
  }

  function renderizarContribuciones(contribuciones, puntajeLr) {
    const listaPos = document.getElementById('lista-factores-pos');
    const listaNeg = document.getElementById('lista-factores-neg');
    listaPos.textContent = '';
    listaNeg.textContent = '';

    const positivos = contribuciones.filter((c) => c.deltaPuntos > 0.05).sort((a, b) => b.deltaPuntos - a.deltaPuntos).slice(0, 5);
    const negativos = contribuciones.filter((c) => c.deltaPuntos < -0.05).sort((a, b) => a.deltaPuntos - b.deltaPuntos).slice(0, 5);
    const maxAbs = Math.max(
      1,
      ...positivos.map((c) => c.deltaPuntos),
      ...negativos.map((c) => Math.abs(c.deltaPuntos))
    );

    const poblar = (ul, items, esPos) => {
      if (!items.length) {
        const li = document.createElement('li');
        li.textContent = 'Sin variaciones relevantes en esta dirección.';
        ul.appendChild(li);
        return;
      }
      for (const item of items) {
        const li = document.createElement('li');
        li.className = 'factor-item';
        const cab = document.createElement('div');
        cab.className = 'factor-texto';
        const lbl = document.createElement('span');
        lbl.textContent = item.etiqueta;
        const pts = document.createElement('span');
        pts.className = 'factor-pts';
        pts.textContent = `${item.deltaPuntos >= 0 ? '+' : ''}${fmtNum.format(item.deltaPuntos)} pts`;
        cab.append(lbl, pts);

        const barraFondo = document.createElement('div');
        barraFondo.className = 'factor-barra-fondo';
        const barra = document.createElement('div');
        barra.className = `factor-barra ${esPos ? 'pos' : 'neg'}`;
        barra.style.width = `${Math.min(100, (Math.abs(item.deltaPuntos) / maxAbs) * 100).toFixed(1)}%`;
        barraFondo.appendChild(barra);

        li.append(cab, barraFondo);
        ul.appendChild(li);
      }
    };

    poblar(listaPos, positivos, true);
    poblar(listaNeg, negativos, false);

    const refCartera = modelo.regresion_logistica.puntaje_cartera_media;
    const diff = puntajeLr - refCartera;
    document.getElementById('resumen-aditivo-lr').textContent =
      `Descomposición exacta de la Regresión Logística respecto a la cartera promedio (${fmtNum.format(refCartera)} pts): ` +
      `este perfil suma ${diff >= 0 ? '+' : ''}${fmtNum.format(diff)} pts para llegar a ${fmtNum.format(puntajeLr)} pts.`;
  }

  function actualizarResultados() {
    const perfil = leerPerfilDesdeFormulario();
    const res = evaluarPerfilCompleto(perfil, modelo);
    ultimoEvaluado = { perfil, res };

    // Gauge
    const minVis = 470;
    const maxVis = 640;
    const frac = Math.max(0, Math.min(1, (res.puntajeMlp - minVis) / (maxVis - minVis)));
    const arco = document.getElementById('gauge-arco');
    arco.style.strokeDashoffset = String((100 * (1 - frac)).toFixed(2));
    arco.style.stroke = res.banda.banda >= 8 ? '#2E6B3F' : res.banda.banda >= 4 ? '#3A5A80' : '#A24C24';

    const txtGauge = document.getElementById('gauge-puntaje');
    txtGauge.textContent = fmtNum.format(res.puntajeMlp);
    document.getElementById('gauge-p0-texto').textContent = `P₀=${Math.round(modelo.scorecard.P0)}`;

    const svgGauge = document.getElementById('svg-gauge');
    svgGauge.setAttribute(
      'aria-label',
      `Puntaje Scorecard ${fmtNum.format(res.puntajeMlp)} puntos, Banda ${res.banda.banda} de 10, probabilidad de incumplimiento ${fmtPct.format(res.pdMlp * 100)} por ciento.`
    );

    // KPIs
    document.getElementById('out-pd-mlp').textContent = `${fmtPct.format(res.pdMlp * 100)} %`;
    const badge = document.getElementById('out-banda-badge');
    badge.className = 'insignia-banda';
    if (res.banda.banda >= 8) {
      badge.classList.add('riesgo-bajo');
      badge.textContent = `Banda D${res.banda.banda} · Riesgo bajo`;
    } else if (res.banda.banda >= 4) {
      badge.classList.add('riesgo-medio');
      badge.textContent = `Banda D${res.banda.banda} · Riesgo moderado`;
    } else {
      badge.classList.add('riesgo-alto');
      badge.textContent = `Banda D${res.banda.banda} · Riesgo alto`;
    }

    document.getElementById('out-percentil').textContent = `Percentil ${fmtNum.format(res.percentil)} (supera al ${fmtNum.format(res.percentil)} %)`;
    document.getElementById('out-tasa-banda').textContent = `${fmtPct.format(res.banda.tasa_malos_observada * 100)} % observada`;
    document.getElementById('out-lr-ref').textContent = `${fmtNum.format(res.puntajeLr)} pts (${fmtPct.format(res.pdLr * 100)} % PD)`;

    // Delta What-If
    const deltaBanner = document.getElementById('badge-delta-whatif');
    const leyendaBase = document.getElementById('leyenda-base-item');
    if (evaluacionBase) {
      const dPts = res.puntajeMlp - evaluacionBase.puntajeMlp;
      const dPdPp = (res.pdMlp - evaluacionBase.pdMlp) * 100;
      const dPct = res.percentil - evaluacionBase.percentil;
      deltaBanner.hidden = false;
      leyendaBase.hidden = false;
      document.getElementById('texto-delta-whatif').textContent =
        `${dPts >= 0 ? '+' : ''}${fmtNum.format(dPts)} pts | PD ${dPdPp >= 0 ? '+' : ''}${fmtPct.format(dPdPp)} pp | percentil ${dPct >= 0 ? '+' : ''}${fmtNum.format(dPct)}`;
    } else {
      deltaBanner.hidden = true;
      leyendaBase.hidden = true;
    }

    construirTablaBandas(res.banda.banda);
    renderizarHistograma(res.puntajeMlp, res.percentil);
    renderizarPalancasMlp(perfil, res.puntajeMlp);
    renderizarContribuciones(res.contribuciones, res.puntajeLr);

    // Tarjeta Parte 2 (Pérdida esperada)
    const cardEl = document.getElementById('tarjeta-perdida-esperada');
    if (res.parte2) {
      cardEl.hidden = false;
      document.getElementById('el-pd').textContent = `${fmtPct.format(res.pdMlp * 100)} %`;
      document.getElementById('el-lgd').textContent = `${fmtPct.format(res.parte2.lgd * 100)} %`;
      document.getElementById('el-ead').textContent = fmtUsd.format(res.parte2.ead);
      document.getElementById('el-ead-frac').textContent = `${fmtPct.format(res.parte2.factorEad * 100)} % del monto (${perfil.plazo_meses} m)`;
      document.getElementById('el-monto').textContent = fmtUsd.format(res.parte2.el);
      document.getElementById('el-pct').textContent = `${fmtPct.format(res.parte2.elPctMonto * 100)} % del monto fondeado`;
    } else {
      cardEl.hidden = true;
    }
  }

  // Subtítulos y metadatos desde el JSON real
  const evalRed = modelo.evaluacion_prueba.red_neuronal;
  const scParams = modelo.scorecard;
  document.getElementById('subtitulo-modelo').textContent =
    `Modelo principal: MLP [${modelo.mlp_pd.capas_ocultas.join(', ')}] ${modelo.mlp_pd.activacion.toUpperCase()} ` +
    `(AUC prueba ${evalRed.auc.toFixed(4)}) · Escala P₀=${Math.round(scParams.P0)}, PDO=${Math.round(scParams.pdo)}`;
  document.getElementById('subtitulo-poblacion').textContent =
    `Distribución real sobre ${modelo.meta.n_prueba.toLocaleString('es-CO')} créditos de prueba (2014)`;
  document.getElementById('pie-meta-modelo').textContent =
    `Exportación: ${modelo.meta.fecha_exportacion} · Semilla MLP: ${modelo.mlp_pd.semilla} · ` +
    `Parámetros: ${modelo.mlp_pd.parametros.toLocaleString('es-CO')} · Actualizable vía python -m src.exportar_app`;

  // Eventos de presets
  const botonesPreset = document.querySelectorAll('.btn-preset');
  const cargarPreset = (id) => {
    const encontrado = modelo.perfiles_ejemplo.find((p) => p.id === id) || modelo.perfiles_ejemplo[1];
    if (!encontrado) return;
    presetActivoId = encontrado.id;
    botonesPreset.forEach((b) => b.classList.toggle('activo', b.dataset.preset === encontrado.id));
    chkAutoCuota.checked = true;
    escribirPerfilEnFormulario(encontrado.valores);
    actualizarResultados();
    anunciador.textContent = `Perfil cargado: ${encontrado.nombre}. Puntaje ${fmtNum.format(ultimoEvaluado.res.puntajeMlp)} puntos.`;
  };

  botonesPreset.forEach((btn) => {
    btn.addEventListener('click', () => cargarPreset(btn.dataset.preset));
  });

  document.getElementById('btn-reiniciar').addEventListener('click', () => {
    cargarPreset(presetActivoId || 'perfil_medio');
  });

  // Eventos del formulario principal
  form.addEventListener('input', () => {
    sincronizarCamposDependientes();
    sincronizarSlidersWhatIf();
    actualizarResultados();
  });

  form.addEventListener('change', () => {
    sincronizarCamposDependientes();
    sincronizarSlidersWhatIf();
    actualizarResultados();
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    validador.syncTodos();
    if (!form.checkValidity()) {
      const primero = form.querySelector(':invalid');
      if (primero) {
        primero.focus();
        anunciador.textContent = 'Hay campos por corregir antes de validar el perfil.';
      }
      return;
    }
    actualizarResultados();
    const res = ultimoEvaluado.res;
    anunciador.textContent =
      `Perfil válido. Puntaje Scorecard ${fmtNum.format(res.puntajeMlp)} puntos, PD ${fmtPct.format(res.pdMlp * 100)} por ciento, Banda D${res.banda.banda}.`;
    document.getElementById('resultados-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  // Sliders y botones What-If sincronizados en dos vías
  for (const [clave, slider] of Object.entries(mapaSliders)) {
    slider.addEventListener('input', () => {
      const el = form.elements.namedItem(clave);
      if (!el) return;
      el.value = slider.value;
      const lblVal = document.getElementById(`val-sld-${clave}`);
      if (lblVal) lblVal.textContent = ETIQUETAS_SLIDERS[clave](Number(slider.value));
      sincronizarCamposDependientes();
      actualizarResultados();
    });
  }

  ['btn-whatif-36', 'btn-whatif-60'].forEach((btnId) => {
    document.getElementById(btnId).addEventListener('click', (e) => {
      const plazo = e.currentTarget.dataset.plazo;
      const radio = form.querySelector(`input[name="plazo_meses"][value="${plazo}"]`);
      if (radio) {
        radio.checked = true;
        sincronizarCamposDependientes();
        sincronizarSlidersWhatIf();
        actualizarResultados();
      }
    });
  });

  // Fijar / restaurar / quitar base What-If
  const btnFijar = document.getElementById('btn-fijar-base');
  const btnRestaurar = document.getElementById('btn-restaurar-base');
  const btnQuitar = document.getElementById('btn-quitar-base');
  const txtBase = document.getElementById('txt-estado-base');

  btnFijar.addEventListener('click', () => {
    perfilBase = leerPerfilDesdeFormulario();
    evaluacionBase = evaluarPerfilCompleto(perfilBase, modelo);
    btnRestaurar.disabled = false;
    btnQuitar.disabled = false;
    txtBase.textContent = `Base: ${fmtNum.format(evaluacionBase.puntajeMlp)} pts (${fmtPct.format(evaluacionBase.pdMlp * 100)} %)`;
    actualizarResultados();
    anunciador.textContent = `Perfil base fijado en ${fmtNum.format(evaluacionBase.puntajeMlp)} puntos.`;
  });

  btnRestaurar.addEventListener('click', () => {
    if (!perfilBase) return;
    escribirPerfilEnFormulario(perfilBase);
    actualizarResultados();
    anunciador.textContent = 'Perfil restaurado a la base fijada.';
  });

  btnQuitar.addEventListener('click', () => {
    perfilBase = null;
    evaluacionBase = null;
    btnRestaurar.disabled = true;
    btnQuitar.disabled = true;
    txtBase.textContent = 'Sin base fijada';
    actualizarResultados();
    anunciador.textContent = 'Comparación contra perfil base desactivada.';
  });

  // Inicialización con perfil promedio
  cargarPreset('perfil_medio');
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', iniciarAplicacion);
  } else {
    iniciarAplicacion();
  }
}
