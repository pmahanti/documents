// Lunar Missions Explorer - Main Application Script
// Works completely offline with embedded mission data

let missionsData = [];
let filteredMissions = [];

// Load missions data
async function loadMissionsData() {
    try {
        const response = await fetch('missions_data.json');
        const data = await response.json();
        missionsData = data.missions;
        filteredMissions = missionsData;
        initializeApp();
    } catch (error) {
        console.error('Error loading missions data:', error);
        document.getElementById('missionsContainer').innerHTML =
            '<div class="no-results"><h3>Error loading mission data</h3><p>Please ensure missions_data.json is in the same directory.</p></div>';
    }
}

// Initialize the application
function initializeApp() {
    populateFilters();
    renderMissions(filteredMissions);
    updateStats();
    attachEventListeners();
}

// Populate filter dropdowns
function populateFilters() {
    const agencies = [...new Set(missionsData.map(m => m.agency))].sort();
    const countries = [...new Set(missionsData.map(m => m.country))].sort();
    const types = [...new Set(missionsData.map(m => m.mission_type))].sort();

    const agencyFilter = document.getElementById('agencyFilter');
    const countryFilter = document.getElementById('countryFilter');
    const typeFilter = document.getElementById('typeFilter');

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
}

// Attach event listeners
function attachEventListeners() {
    document.getElementById('searchInput').addEventListener('input', handleFilters);
    document.getElementById('statusFilter').addEventListener('change', handleFilters);
    document.getElementById('agencyFilter').addEventListener('change', handleFilters);
    document.getElementById('countryFilter').addEventListener('change', handleFilters);
    document.getElementById('typeFilter').addEventListener('change', handleFilters);

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

    filteredMissions = missionsData.filter(mission => {
        // Search filter
        const matchesSearch = searchTerm === '' ||
            mission.name.toLowerCase().includes(searchTerm) ||
            mission.description.toLowerCase().includes(searchTerm) ||
            mission.agency.toLowerCase().includes(searchTerm) ||
            mission.landing_site.toLowerCase().includes(searchTerm) ||
            mission.payloads.some(p =>
                p.name.toLowerCase().includes(searchTerm) ||
                p.type.toLowerCase().includes(searchTerm) ||
                p.description.toLowerCase().includes(searchTerm)
            );

        // Status filter
        const matchesStatus = statusFilter === 'all' || mission.status === statusFilter;

        // Agency filter
        const matchesAgency = agencyFilter === 'all' || mission.agency === agencyFilter;

        // Country filter
        const matchesCountry = countryFilter === 'all' || mission.country === countryFilter;

        // Type filter
        const matchesType = typeFilter === 'all' || mission.mission_type === typeFilter;

        return matchesSearch && matchesStatus && matchesAgency && matchesCountry && matchesType;
    });

    renderMissions(filteredMissions);
    updateStats();
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

    return `
        <div class="mission-card" data-mission-id="${mission.id}">
            <div class="mission-header">
                <h2 class="mission-name">${mission.name}</h2>
                <p class="mission-agency">${mission.agency} • ${mission.country}</p>
            </div>

            <div class="mission-badges">
                <span class="badge badge-status ${mission.status}">${mission.status.toUpperCase()}</span>
                <span class="badge badge-type">${mission.mission_type}</span>
            </div>

            <div class="mission-details">
                <p><strong>Launch:</strong> ${launchDate}</p>
                <p><strong>Landing:</strong> ${landingDate}</p>
                <p><strong>Site:</strong> ${mission.landing_site}</p>
            </div>

            <div class="mission-description">
                ${mission.description}
            </div>

            <div class="payload-count">
                ${mission.payloads.length} Payload${mission.payloads.length !== 1 ? 's' : ''} / Instrument${mission.payloads.length !== 1 ? 's' : ''}
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

    modalBody.innerHTML = `
        <div class="modal-mission-header">
            <h2 class="modal-mission-name">${mission.name}</h2>
            <div class="mission-badges">
                <span class="badge badge-status ${mission.status}">${mission.status.toUpperCase()}</span>
                <span class="badge badge-type">${mission.mission_type}</span>
            </div>
        </div>

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
            <div class="info-item">
                <strong>Mission Type</strong>
                <span>${mission.mission_type}</span>
            </div>
        </div>

        <div class="mission-description">
            <h3>Mission Overview</h3>
            <p>${mission.description}</p>
        </div>

        <div class="payloads-section">
            <h3>Scientific Payloads & Instruments (${mission.payloads.length})</h3>
            ${mission.payloads.map(payload => createPayloadCard(payload)).join('')}
        </div>
    `;

    modal.style.display = 'block';
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

    return `
        <div class="payload-card">
            <div class="payload-name">${payload.name}</div>
            <div class="payload-type">${payload.type}</div>
            <div class="payload-description">${payload.description}</div>
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

    stats.innerHTML = `
        Showing ${total} mission${total !== 1 ? 's' : ''}
        (${past} past, ${present} present, ${future} future)
        with ${totalPayloads} total payload${totalPayloads !== 1 ? 's' : ''}/instrument${totalPayloads !== 1 ? 's' : ''}
    `;
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', loadMissionsData);
