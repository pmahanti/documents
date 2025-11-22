// Lunar Missions Explorer - Enhanced Application Script
// Version 2.0 with cross-cutting analysis, vendors, costs, and publications

let missionsData = [];
let filteredMissions = [];
let metadata = {};
let instrumentCategories = [];
let scienceObjectives = [];
let currentView = 'missions'; // 'missions', 'instruments', 'science'

// Load missions data
async function loadMissionsData() {
    try {
        const response = await fetch('missions_data_enhanced.json');
        const data = await response.json();
        metadata = data.metadata || {};
        instrumentCategories = data.instrument_categories || [];
        scienceObjectives = data.science_objectives || [];
        missionsData = data.missions;
        filteredMissions = missionsData;
        initializeApp();
    } catch (error) {
        console.error('Error loading missions data:', error);
        document.getElementById('missionsContainer').innerHTML =
            '<div class="no-results"><h3>Error loading mission data</h3><p>Please ensure missions_data_enhanced.json is in the same directory.</p></div>';
    }
}

// Initialize the application
function initializeApp() {
    populateFilters();
    renderContent();
    updateStats();
    attachEventListeners();
}

// Populate filter dropdowns
function populateFilters() {
    const agencies = [...new Set(missionsData.map(m => m.agency))].sort();
    const countries = [...new Set(missionsData.map(m => m.country))].sort();
    const types = [...new Set(missionsData.map(m => m.mission_type))].sort();
    const outcomes = [...new Set(missionsData.map(m => m.outcome))].filter(Boolean).sort();

    const agencyFilter = document.getElementById('agencyFilter');
    const countryFilter = document.getElementById('countryFilter');
    const typeFilter = document.getElementById('typeFilter');
    const outcomeFilter = document.getElementById('outcomeFilter');

    agencies.forEach(agency => {
        const option = document.createElement('option');
        option.value = agency;
        option.textContent = agency;
        agencyFilter.appendChild(option);
    });

    countries.forEach(country => {
        const option = document.createElement('option');
        option.value = country;
        option.textContent = country;
        countryFilter.appendChild(option);
    });

    types.forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.textContent = type;
        typeFilter.appendChild(option);
    });

    outcomes.forEach(outcome => {
        const option = document.createElement('option');
        option.value = outcome;
        option.textContent = outcome.replace('_', ' ').toUpperCase();
        outcomeFilter.appendChild(option);
    });

    // Populate instrument and science filters
    const instrumentFilter = document.getElementById('instrumentFilter');
    const scienceFilter = document.getElementById('scienceFilter');

    instrumentCategories.forEach(cat => {
        const option = document.createElement('option');
        option.value = cat;
        option.textContent = cat;
        instrumentFilter.appendChild(option);
    });

    scienceObjectives.forEach(obj => {
        const option = document.createElement('option');
        option.value = obj;
        option.textContent = obj;
        scienceFilter.appendChild(option);
    });
}

// Attach event listeners
function attachEventListeners() {
    document.getElementById('searchInput').addEventListener('input', handleFilters);
    document.getElementById('statusFilter').addEventListener('change', handleFilters);
    document.getElementById('agencyFilter').addEventListener('change', handleFilters);
    document.getElementById('countryFilter').addEventListener('change', handleFilters);
    document.getElementById('typeFilter').addEventListener('change', handleFilters);
    document.getElementById('outcomeFilter').addEventListener('change', handleFilters);
    document.getElementById('instrumentFilter').addEventListener('change', handleFilters);
    document.getElementById('scienceFilter').addEventListener('change', handleFilters);

    // View switchers
    document.querySelectorAll('.view-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            currentView = e.target.dataset.view;
            document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            renderContent();
        });
    });

    // Modal close handlers
    const modal = document.getElementById('missionModal');
    const closeBtn = document.querySelector('.close');

    closeBtn.onclick = () => modal.style.display = 'none';
    window.onclick = (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    };
}

// Handle all filters
function handleFilters() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const statusFilter = document.getElementById('statusFilter').value;
    const agencyFilter = document.getElementById('agencyFilter').value;
    const countryFilter = document.getElementById('countryFilter').value;
    const typeFilter = document.getElementById('typeFilter').value;
    const outcomeFilter = document.getElementById('outcomeFilter').value;
    const instrumentFilter = document.getElementById('instrumentFilter').value;
    const scienceFilter = document.getElementById('scienceFilter').value;

    filteredMissions = missionsData.filter(mission => {
        // Search filter
        const matchesSearch = searchTerm === '' ||
            mission.name.toLowerCase().includes(searchTerm) ||
            mission.description.toLowerCase().includes(searchTerm) ||
            mission.agency.toLowerCase().includes(searchTerm) ||
            mission.landing_site.toLowerCase().includes(searchTerm) ||
            (mission.failure_reason && mission.failure_reason.toLowerCase().includes(searchTerm)) ||
            mission.payloads.some(p =>
                p.name.toLowerCase().includes(searchTerm) ||
                p.type.toLowerCase().includes(searchTerm) ||
                p.description.toLowerCase().includes(searchTerm) ||
                (p.vendor && p.vendor.toLowerCase().includes(searchTerm)) ||
                (p.core_science && p.core_science.toLowerCase().includes(searchTerm))
            );

        // Status filter
        const matchesStatus = statusFilter === 'all' || mission.status === statusFilter;

        // Agency filter
        const matchesAgency = agencyFilter === 'all' || mission.agency === agencyFilter;

        // Country filter
        const matchesCountry = countryFilter === 'all' || mission.country === countryFilter;

        // Type filter
        const matchesType = typeFilter === 'all' || mission.mission_type === typeFilter;

        // Outcome filter
        const matchesOutcome = outcomeFilter === 'all' || mission.outcome === outcomeFilter;

        // Instrument category filter
        const matchesInstrument = instrumentFilter === 'all' ||
            mission.payloads.some(p => p.instrument_category === instrumentFilter);

        // Science objective filter
        const matchesScience = scienceFilter === 'all' ||
            mission.payloads.some(p => p.science_objectives && p.science_objectives.includes(scienceFilter));

        return matchesSearch && matchesStatus && matchesAgency && matchesCountry &&
               matchesType && matchesOutcome && matchesInstrument && matchesScience;
    });

    renderContent();
    updateStats();
}

// Render content based on current view
function renderContent() {
    switch (currentView) {
        case 'missions':
            renderMissions(filteredMissions);
            break;
        case 'instruments':
            renderInstrumentAnalysis();
            break;
        case 'science':
            renderScienceAnalysis();
            break;
    }
}

// Render missions
function renderMissions(missions) {
    const container = document.getElementById('missionsContainer');

    if (missions.length === 0) {
        container.innerHTML = `
            <div class="no-results">
                <h3>No missions found</h3>
                <p>Try adjusting your search or filters</p>
            </div>
        `;
        return;
    }

    container.innerHTML = missions.map(mission => createMissionCard(mission)).join('');

    // Attach click handlers to mission cards
    document.querySelectorAll('.mission-card').forEach(card => {
        card.addEventListener('click', () => {
            const missionId = card.dataset.missionId;
            const mission = missionsData.find(m => m.id === missionId);
            showMissionModal(mission);
        });
    });
}

// Create mission card HTML
function createMissionCard(mission) {
    const launchDate = new Date(mission.launch_date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });

    const landingDate = mission.landing_date ? new Date(mission.landing_date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    }) : 'TBD';

    const outcomeClass = mission.outcome === 'failure' ? 'failure' :
                         mission.outcome === 'partial_success' ? 'partial' : 'success';

    const costInfo = mission.mission_cost_usd ?
        `<p><strong>Cost:</strong> $${(mission.mission_cost_usd / 1000000).toFixed(1)}M</p>` : '';

    const failureInfo = mission.failure_reason ?
        `<div class="failure-reason"><strong>⚠️ Failure:</strong> ${mission.failure_reason}</div>` : '';

    return `
        <div class="mission-card ${outcomeClass}" data-mission-id="${mission.id}">
            <div class="mission-header">
                <h2 class="mission-name">${mission.name}</h2>
                <p class="mission-agency">${mission.agency} • ${mission.country}</p>
            </div>

            <div class="mission-badges">
                <span class="badge badge-status ${mission.status}">${mission.status.toUpperCase()}</span>
                <span class="badge badge-type">${mission.mission_type}</span>
                ${mission.outcome ? `<span class="badge badge-outcome ${outcomeClass}">${mission.outcome.replace('_', ' ').toUpperCase()}</span>` : ''}
            </div>

            <div class="mission-details">
                <p><strong>Launch:</strong> ${launchDate}</p>
                <p><strong>Landing:</strong> ${landingDate}</p>
                <p><strong>Site:</strong> ${mission.landing_site}</p>
                ${costInfo}
            </div>

            ${failureInfo}

            <div class="mission-description">
                ${mission.description}
            </div>

            <div class="payload-count">
                ${mission.payloads.length} Payload${mission.payloads.length !== 1 ? 's' : ''} / Instrument${mission.payloads.length !== 1 ? 's' : ''}
            </div>
        </div>
    `;
}

// Render instrument cross-cutting analysis
function renderInstrumentAnalysis() {
    const container = document.getElementById('missionsContainer');

    // Group instruments by category
    const instrumentsByCategory = {};
    filteredMissions.forEach(mission => {
        mission.payloads.forEach(payload => {
            const category = payload.instrument_category || 'Other';
            if (!instrumentsByCategory[category]) {
                instrumentsByCategory[category] = [];
            }
            instrumentsByCategory[category].push({
                ...payload,
                mission: mission.name,
                missionId: mission.id,
                outcome: mission.outcome
            });
        });
    });

    const sortedCategories = Object.keys(instrumentsByCategory).sort();

    container.innerHTML = `
        <div class="analysis-view">
            <h2>Cross-Cutting Analysis: By Instrument Type</h2>
            <p class="analysis-desc">Explore all lunar instruments grouped by their type across all missions</p>
            ${sortedCategories.map(category => createInstrumentCategorySection(category, instrumentsByCategory[category])).join('')}
        </div>
    `;
}

// Create instrument category section
function createInstrumentCategorySection(category, instruments) {
    const successCount = instruments.filter(i => i.outcome === 'success').length;
    const failureCount = instruments.filter(i => i.outcome === 'failure').length;

    return `
        <div class="category-section">
            <h3 class="category-title">
                ${category}
                <span class="category-count">(${instruments.length} instruments)</span>
                <span class="category-stats">✓ ${successCount} | ✗ ${failureCount}</span>
            </h3>
            <div class="instrument-grid">
                ${instruments.map(inst => createInstrumentCard(inst)).join('')}
            </div>
        </div>
    `;
}

// Create instrument card
function createInstrumentCard(instrument) {
    const vendor = instrument.vendor ? `<p><strong>Vendor:</strong> ${instrument.vendor}</p>` : '';
    const cost = instrument.cost_usd ? `<p><strong>Cost:</strong> $${(instrument.cost_usd / 1000000).toFixed(1)}M</p>` : '';
    const coreScience = instrument.core_science ? `<p class="core-science"><strong>Science:</strong> ${instrument.core_science}</p>` : '';
    const outcomeIcon = instrument.outcome === 'success' ? '✓' :
                        instrument.outcome === 'failure' ? '✗' :
                        instrument.outcome === 'partial_success' ? '◐' : '○';

    return `
        <div class="instrument-card">
            <div class="instrument-name">
                <span class="outcome-icon ${instrument.outcome}">${outcomeIcon}</span>
                ${instrument.name}
            </div>
            <div class="instrument-mission">${instrument.mission}</div>
            ${vendor}
            ${cost}
            <p>${instrument.description}</p>
            ${coreScience}
            ${instrument.science_objectives ? `
                <div class="science-tags">
                    ${instrument.science_objectives.map(obj => `<span class="science-tag">${obj}</span>`).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

// Render science objective cross-cutting analysis
function renderScienceAnalysis() {
    const container = document.getElementById('missionsContainer');

    // Group instruments by science objective
    const instrumentsByScience = {};
    filteredMissions.forEach(mission => {
        mission.payloads.forEach(payload => {
            if (payload.science_objectives) {
                payload.science_objectives.forEach(objective => {
                    if (!instrumentsByScience[objective]) {
                        instrumentsByScience[objective] = [];
                    }
                    instrumentsByScience[objective].push({
                        ...payload,
                        mission: mission.name,
                        missionId: mission.id,
                        outcome: mission.outcome
                    });
                });
            }
        });
    });

    const sortedObjectives = Object.keys(instrumentsByScience).sort();

    container.innerHTML = `
        <div class="analysis-view">
            <h2>Cross-Cutting Analysis: By Science Objective</h2>
            <p class="analysis-desc">Explore all instruments grouped by their primary science objectives</p>
            ${sortedObjectives.map(objective => createScienceObjectiveSection(objective, instrumentsByScience[objective])).join('')}
        </div>
    `;
}

// Create science objective section
function createScienceObjectiveSection(objective, instruments) {
    const uniqueInstruments = instruments.filter((inst, index, self) =>
        index === self.findIndex(i => i.name === inst.name && i.mission === inst.mission)
    );

    const missions = [...new Set(instruments.map(i => i.mission))];

    return `
        <div class="category-section">
            <h3 class="category-title">
                ${objective}
                <span class="category-count">(${uniqueInstruments.length} instruments across ${missions.length} missions)</span>
            </h3>
            <div class="instrument-grid">
                ${uniqueInstruments.map(inst => createInstrumentCard(inst)).join('')}
            </div>
        </div>
    `;
}

// Show mission modal with full details
function showMissionModal(mission) {
    const modal = document.getElementById('missionModal');
    const modalBody = document.getElementById('modalBody');

    const launchDate = new Date(mission.launch_date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    const landingDate = mission.landing_date ? new Date(mission.landing_date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    }) : 'To Be Determined';

    const outcomeClass = mission.outcome === 'failure' ? 'failure' :
                         mission.outcome === 'partial_success' ? 'partial' : 'success';

    const costInfo = mission.mission_cost_usd ?
        `<div class="info-item">
            <strong>Mission Cost</strong>
            <span>$${(mission.mission_cost_usd / 1000000).toFixed(0)} Million</span>
        </div>` : '';

    const coordinates = mission.coordinates ?
        `<div class="info-item">
            <strong>Coordinates</strong>
            <span>${mission.coordinates.lat.toFixed(2)}°, ${mission.coordinates.lon.toFixed(2)}°</span>
        </div>` : '';

    const failureInfo = mission.failure_reason ?
        `<div class="failure-detail">
            <h3>⚠️ Mission Outcome: ${mission.outcome.replace('_', ' ').toUpperCase()}</h3>
            <p><strong>Reason:</strong> ${mission.failure_reason}</p>
        </div>` : '';

    const publicationsSection = mission.publications && mission.publications.length > 0 ?
        `<div class="publications-section">
            <h3>📚 Publications & Reports</h3>
            ${mission.publications.map(pub => createPublicationCard(pub)).join('')}
        </div>` : '';

    modalBody.innerHTML = `
        <div class="modal-mission-header">
            <h2 class="modal-mission-name">${mission.name}</h2>
            <div class="mission-badges">
                <span class="badge badge-status ${mission.status}">${mission.status.toUpperCase()}</span>
                <span class="badge badge-type">${mission.mission_type}</span>
                ${mission.outcome ? `<span class="badge badge-outcome ${outcomeClass}">${mission.outcome.replace('_', ' ').toUpperCase()}</span>` : ''}
            </div>
        </div>

        ${failureInfo}

        <div class="modal-mission-info">
            <div class="info-item">
                <strong>Agency</strong>
                <span>${mission.agency}</span>
            </div>
            <div class="info-item">
                <strong>Country</strong>
                <span>${mission.country}</span>
            </div>
            <div class="info-item">
                <strong>Launch Date</strong>
                <span>${launchDate}</span>
            </div>
            <div class="info-item">
                <strong>Landing Date</strong>
                <span>${landingDate}</span>
            </div>
            <div class="info-item">
                <strong>Landing Site</strong>
                <span>${mission.landing_site}</span>
            </div>
            ${coordinates}
            ${costInfo}
        </div>

        <div class="mission-description">
            <h3>Mission Overview</h3>
            <p>${mission.description}</p>
        </div>

        ${publicationsSection}

        <div class="payloads-section">
            <h3>Scientific Payloads & Instruments (${mission.payloads.length})</h3>
            ${mission.payloads.map(payload => createPayloadCard(payload)).join('')}
        </div>
    `;

    modal.style.display = 'block';
}

// Create publication card
function createPublicationCard(pub) {
    const citation = pub.journal ?
        `${pub.journal}${pub.volume ? ` ${pub.volume}` : ''}${pub.pages ? `:${pub.pages}` : ''} (${pub.year || 'N/A'})` :
        pub.type || 'Publication';

    const link = pub.url ?
        `<a href="${pub.url}" target="_blank" class="pub-link">View Publication →</a>` : '';

    return `
        <div class="publication-card">
            <div class="pub-title">${pub.title}</div>
            ${pub.authors ? `<div class="pub-authors">${pub.authors.join(', ')}</div>` : ''}
            <div class="pub-citation">${citation}</div>
            ${pub.findings ? `<div class="pub-findings"><strong>Key Findings:</strong> ${pub.findings}</div>` : ''}
            ${link}
        </div>
    `;
}

// Create payload card HTML
function createPayloadCard(payload) {
    let specsHtml = '';

    if (payload.specifications) {
        specsHtml += `<div class="payload-specs"><strong>Specifications:</strong> ${payload.specifications}</div>`;
    }

    if (payload.mass_kg) {
        specsHtml += `<div class="payload-specs"><strong>Mass:</strong> ${payload.mass_kg} kg</div>`;
    }

    if (payload.power_watts) {
        specsHtml += `<div class="payload-specs"><strong>Power:</strong> ${payload.power_watts} watts</div>`;
    }

    if (payload.vendor) {
        specsHtml += `<div class="payload-specs"><strong>Vendor/Manufacturer:</strong> ${payload.vendor}</div>`;
    }

    if (payload.pi_name || payload.pi_institution) {
        const pi = payload.pi_name || '';
        const inst = payload.pi_institution || '';
        specsHtml += `<div class="payload-specs"><strong>Principal Investigator:</strong> ${pi}${pi && inst ? ', ' : ''}${inst}</div>`;
    }

    if (payload.cost_usd) {
        specsHtml += `<div class="payload-specs"><strong>Cost:</strong> $${(payload.cost_usd / 1000000).toFixed(1)} Million</div>`;
    }

    if (payload.core_science) {
        specsHtml += `<div class="payload-core-science"><strong>Core Science:</strong> ${payload.core_science}</div>`;
    }

    if (payload.key_findings) {
        specsHtml += `<div class="payload-findings"><strong>🔬 Key Findings:</strong> ${payload.key_findings}</div>`;
    }

    if (payload.innovation) {
        specsHtml += `<div class="payload-innovation"><strong>💡 Innovation:</strong> ${payload.innovation}</div>`;
    }

    const scienceObjectives = payload.science_objectives ?
        `<div class="science-tags">
            ${payload.science_objectives.map(obj => `<span class="science-tag">${obj}</span>`).join('')}
        </div>` : '';

    return `
        <div class="payload-card">
            <div class="payload-header">
                <div class="payload-name">${payload.name}</div>
                <div class="payload-type">${payload.type}</div>
            </div>
            <div class="payload-description">${payload.description}</div>
            ${scienceObjectives}
            ${specsHtml}
        </div>
    `;
}

// Update statistics
function updateStats() {
    const stats = document.getElementById('stats');
    const total = filteredMissions.length;
    const past = filteredMissions.filter(m => m.status === 'past').length;
    const present = filteredMissions.filter(m => m.status === 'present').length;
    const future = filteredMissions.filter(m => m.status === 'future').length;
    const totalPayloads = filteredMissions.reduce((sum, m) => sum + m.payloads.length, 0);
    const success = filteredMissions.filter(m => m.outcome === 'success').length;
    const failure = filteredMissions.filter(m => m.outcome === 'failure').length;
    const partial = filteredMissions.filter(m => m.outcome === 'partial_success').length;

    const totalCost = filteredMissions
        .filter(m => m.mission_cost_usd)
        .reduce((sum, m) => sum + m.mission_cost_usd, 0);

    const costInfo = totalCost > 0 ? ` | Total Cost: $${(totalCost / 1000000000).toFixed(1)}B` : '';

    stats.innerHTML = `
        Showing ${total} mission${total !== 1 ? 's' : ''}
        (${past} past, ${present} present, ${future} future)
        with ${totalPayloads} total payload${totalPayloads !== 1 ? 's' : ''}
        | Outcomes: ✓ ${success} | ✗ ${failure} | ◐ ${partial}${costInfo}
    `;
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', loadMissionsData);
