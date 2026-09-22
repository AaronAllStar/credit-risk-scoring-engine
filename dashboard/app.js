// Detección automática del backend: si se abre directamente por archivo local (file://), conecta a http://127.0.0.1:3000
const API_BASE = (window.location.protocol === "file:" || !window.location.port) 
    ? "http://127.0.0.1:3000" 
    : window.location.origin;

let currentPage = 1;
const pageLimit = 10;
let currentSearch = "";

document.addEventListener("DOMContentLoaded", () => {
    initSystem();
    setupEventListeners();
});

async function initSystem() {
    await fetchHealth();
    await fetchCustomers(currentPage);
    // Simular un primer cliente por defecto
    evaluateForm();
}

function setupEventListeners() {
    document.getElementById("score-form").addEventListener("submit", (e) => {
        e.preventDefault();
        evaluateForm();
    });

    const searchInput = document.getElementById("cust-search");
    let debounceTimer;
    searchInput.addEventListener("input", (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            currentSearch = e.target.value.trim();
            currentPage = 1;
            fetchCustomers(currentPage);
        }, 300);
    });

    document.getElementById("btn-prev").addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            fetchCustomers(currentPage);
        }
    });

    document.getElementById("btn-next").addEventListener("click", () => {
        currentPage++;
        fetchCustomers(currentPage);
    });
}

async function fetchHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            const data = await res.json();
            document.getElementById("service-status").textContent = `${data.service} • Activo`;
            document.getElementById("kpi-total-customers").textContent = Number(data.total_customers).toLocaleString();
            document.getElementById("kpi-active-model").textContent = data.active_model.split(" ")[0] + " Forest";
        }
    } catch (e) {
        console.warn("API de salud no respondió:", e);
    }
}

async function fetchCustomers(page) {
    try {
        let url = `${API_BASE}/customers?page=${page}&limit=${pageLimit}`;
        if (currentSearch) {
            url += `&search=${encodeURIComponent(currentSearch)}`;
        }
        const res = await fetch(url);
        if (!res.ok) return;

        const data = await res.json();
        renderCustomersTable(data.items);
        document.getElementById("page-info").textContent = `Mostrando ${data.items.length} de ${data.total} clientes (Página ${data.page})`;
    } catch (e) {
        console.error("Error al cargar clientes:", e);
    }
}

function renderCustomersTable(customers) {
    const tbody = document.getElementById("customers-table-body");
    tbody.innerHTML = "";

    if (!customers || customers.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No se encontraron clientes</td></tr>`;
        return;
    }

    customers.forEach(c => {
        const tr = document.createElement("tr");

        let badgeClass = "badge-green";
        if (c.risk_band === "Alto Riesgo") badgeClass = "badge-red";
        else if (c.risk_band === "Moderado") badgeClass = "badge-amber";

        tr.innerHTML = `
            <td><strong>#${c.customer_id}</strong></td>
            <td>${c.age.toFixed(1)} años</td>
            <td>$${Math.round(c.annual_income).toLocaleString()}</td>
            <td>${(c.delinquency_rate * 100).toFixed(1)}%</td>
            <td><strong style="font-size: 1rem;">${c.credit_score}</strong></td>
            <td><span class="kpi-badge ${badgeClass}">${c.risk_band}</span></td>
            <td><strong>${c.decision}</strong></td>
            <td>
                <button class="btn-primary" style="padding: 4px 10px; font-size: 0.75rem;" onclick="loadCustomerToSimulator(${c.customer_id})">
                    Inspeccionar
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

async function loadCustomerToSimulator(id) {
    try {
        const res = await fetch(`${API_BASE}/customers/${id}`);
        if (!res.ok) return;
        const data = await res.json();
        renderScoreResult(data);
    } catch (e) {
        console.error("Error al obtener detalle del cliente:", e);
    }
}

async function evaluateForm() {
    const payload = {
        age_years: parseFloat(document.getElementById("inp-age").value),
        employment_years: parseFloat(document.getElementById("inp-employment").value),
        annual_income: parseFloat(document.getElementById("inp-income").value),
        children_count: parseFloat(document.getElementById("inp-children").value),
        family_members_count: parseFloat(document.getElementById("inp-family").value),
        credit_history_length_months: parseFloat(document.getElementById("inp-history-len").value),
        historical_delinquency_rate: parseFloat(document.getElementById("inp-delinq-rate").value),
        max_past_due_severity: parseFloat(document.getElementById("inp-max-sev").value),
        own_realty: parseFloat(document.getElementById("inp-realty").value),
        own_car: parseFloat(document.getElementById("inp-car").value),
        has_work_phone: 1.0,
        has_email: 1.0,
        delinquency_rate_last_6m: parseFloat(document.getElementById("inp-delinq-rate").value)
    };

    try {
        const res = await fetch(`${API_BASE}/score`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            const data = await res.json();
            renderScoreResult(data);
        }
    } catch (e) {
        console.error("Error al calcular score:", e);
    }
}

function renderScoreResult(data) {
    const scoreElem = document.getElementById("res-score");
    const bandElem = document.getElementById("res-band");
    const probElem = document.getElementById("res-prob");
    const badgeElem = document.getElementById("decision-badge");
    const arc = document.getElementById("score-arc");

    scoreElem.textContent = data.credit_score;
    probElem.textContent = (data.probability_of_default * 100).toFixed(2) + "%";

    // Mapeo de score (300 a 850) a longitud del arco SVG (251.2 es el perímetro)
    const norm = Math.max(0, Math.min(1, (data.credit_score - 300) / 550));
    const offset = 251.2 * (1 - norm);
    arc.style.strokeDashoffset = offset;

    let color = "#10b981"; // Emerald
    let badgeText = "APROBADO";
    let badgeClass = "badge-green";

    if (data.credit_score < 580) {
        color = "#f43f5e";
        badgeText = "RECHAZADO";
        badgeClass = "badge-red";
        bandElem.textContent = "Alto Riesgo • Deficiente";
    } else if (data.credit_score < 670) {
        color = "#f59e0b";
        badgeText = "REVISIÓN MANUAL";
        badgeClass = "badge-amber";
        bandElem.textContent = "Riesgo Moderado • Fair";
    } else if (data.credit_score < 750) {
        color = "#3b82f6";
        badgeText = "APROBADO";
        badgeClass = "badge-blue";
        bandElem.textContent = "Bajo Riesgo • Bueno";
    } else {
        color = "#10b981";
        badgeText = "APROBADO";
        badgeClass = "badge-green";
        bandElem.textContent = "Riesgo Mínimo • Excelente";
    }

    arc.style.stroke = color;
    bandElem.style.color = color;
    badgeElem.textContent = badgeText;
    badgeElem.className = `kpi-badge ${badgeClass}`;

    // Render Waterfall Explainability
    const waterfall = document.getElementById("waterfall-list");
    waterfall.innerHTML = "";

    const topFactors = data.factor_contributions.slice(0, 5);
    topFactors.forEach(f => {
        const item = document.createElement("div");
        item.className = "waterfall-item";

        const isPositive = f.impact_direction === "Positivo";
        const fillClass = isPositive ? "fill-positive" : "fill-negative";
        const widthPct = Math.round(f.relative_importance * 100 * 3.5).toString().slice(0, 3);

        item.innerHTML = `
            <div style="font-weight: 500; font-size: 0.8rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                ${f.display_name}
            </div>
            <div class="waterfall-bar-bg">
                <div class="waterfall-bar-fill ${fillClass}" style="width: ${Math.min(100, Math.max(15, widthPct))}%;"></div>
            </div>
            <div style="text-align: right; font-size: 0.75rem; color: ${isPositive ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; font-weight: 600;">
                ${isPositive ? '+' : '-'}${f.display_name.split(' ')[0]}
            </div>
        `;
        waterfall.appendChild(item);
    });
}
