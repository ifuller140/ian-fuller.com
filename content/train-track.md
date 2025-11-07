---
title: 'Live Train Tracking and Booking Website'
description: 'Full-stack web application for Amtrak train booking and management'
image: 'train-track/train-track.jpg'
preview: 'train-track/preview.gif'
priority: 4
tags:
  - JavaScript
  - HTML/CSS
  - REST API
  - Web Development
links:
  - text: View on GitHub
    href: https://github.com/ifuller140/train-tracker
---

## Project Overview

For my web development course, I built a full-stack train booking system that interfaces with the **Amtrak API** to provide real-time train tracking, schedule management, and passenger booking capabilities. This project demonstrates practical application of modern web technologies: API integration, dynamic DOM manipulation, and responsive design—all implemented from scratch without frameworks.

While the visual design is simple and functional (typical of a course project), the **underlying architecture is robust**, handling real-time data updates, complex API interactions, and multi-step user workflows.

![Application Interface](/train-track/app-interface.png)
_Main application interface showing live train status_

---

## Core Functionality

### 1. Live Train Tracking

Query real-time Amtrak train locations and status:

- Current position (lat/long coordinates)
- Estimated arrival times for upcoming stations
- Delay information
- Train speed and direction

**API Integration**:

```javascript
async function getTrainStatus(trainNumber) {
  const response = await fetch(
    `https://api-v3.amtraker.com/v3/trains/${trainNumber}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`Train ${trainNumber} not found`);
  }

  const data = await response.json();
  return parseTrainData(data);
}

function parseTrainData(apiData) {
  return {
    trainNumber: apiData.trainNum,
    routeName: apiData.routeName,
    currentStation: apiData.lastStation,
    nextStation: apiData.stations[0],
    estimatedArrival: apiData.stations[0].arr,
    delay: apiData.stations[0].delay,
    coordinates: {
      lat: apiData.lat,
      lon: apiData.lon,
    },
  };
}
```

### 2. Passenger Management System

CRUD operations for passenger records:

- **Create**: Add new passengers with name, seat assignment, travel dates
- **Read**: View all passengers on a specific train
- **Update**: Modify passenger details or transfer to different train
- **Delete**: Remove passenger bookings

**Data Structure**:

```javascript
const passengers = [
  {
    id: 'P001',
    name: 'John Doe',
    trainNumber: '91',
    origin: 'New York Penn Station',
    destination: 'Washington Union Station',
    seatNumber: '12A',
    departureDate: '2025-01-15',
    ticketClass: 'business',
  },
  // ... more passengers
];
```

### 3. Route Visualization

Display train routes with all station stops:

- Station names and arrival/departure times
- Platform numbers
- Dwell time at each station
- Highlight current station and next destination

**DOM Generation**:

```javascript
function displayRoute(stations) {
  const routeContainer = document.getElementById('route-display');
  routeContainer.innerHTML = ''; // Clear previous

  stations.forEach((station, index) => {
    const stationCard = document.createElement('div');
    stationCard.className = 'station-card';

    // Highlight current station
    if (station.status === 'current') {
      stationCard.classList.add('current-station');
    }

    stationCard.innerHTML = `
            <div class="station-header">
                <span class="station-number">${index + 1}</span>
                <h3>${station.name}</h3>
            </div>
            <div class="station-times">
                <span>Arrival: ${formatTime(station.arr)}</span>
                <span>Departure: ${formatTime(station.dep)}</span>
            </div>
            <div class="station-platform">
                Platform ${station.platform || 'TBD'}
            </div>
        `;

    routeContainer.appendChild(stationCard);
  });
}
```

---

## Technical Implementation

### Architecture Overview

```
┌──────────────────────────────────────────────┐
│           User Interface (HTML/CSS)          │
│  - Train search form                         │
│  - Passenger booking interface               │
│  - Live status display                       │
└───────────────┬──────────────────────────────┘
                │
┌───────────────▼──────────────────────────────┐
│     JavaScript Application Logic             │
│  - Event handlers                            │
│  - Data processing                           │
│  - UI updates                                │
└───────────────┬──────────────────────────────┘
                │
┌───────────────▼──────────────────────────────┐
│          API Service Layer                   │
│  - Amtrak API integration                    │
│  - Error handling                            │
│  - Data caching                              │
└──────────────────────────────────────────────┘
```

### API Integration Strategy

The Amtrak API provides extensive data, but rate limits require careful management:

**Caching Strategy**:

```javascript
const cache = {
  trains: {},
  stations: {},
  expirationTime: 60000, // 1 minute
};

async function fetchWithCache(endpoint, cacheKey) {
  // Check if cached and not expired
  if (
    cache[cacheKey] &&
    Date.now() - cache[cacheKey].timestamp < cache.expirationTime
  ) {
    return cache[cacheKey].data;
  }

  // Fetch fresh data
  const data = await fetch(endpoint).then((r) => r.json());

  // Update cache
  cache[cacheKey] = {
    data: data,
    timestamp: Date.now(),
  };

  return data;
}
```

**Error Handling**:

```javascript
async function safeApiCall(apiFunction, errorMessage) {
  try {
    return await apiFunction();
  } catch (error) {
    console.error(errorMessage, error);
    displayError(errorMessage);
    return null;
  }
}
```

### Data Persistence

Since this is a client-side application, passenger data is stored in **localStorage**:

```javascript
function savePassengers() {
  localStorage.setItem('passengers', JSON.stringify(passengers));
}

function loadPassengers() {
  const stored = localStorage.getItem('passengers');
  return stored ? JSON.parse(stored) : [];
}

// Auto-save on every modification
function addPassenger(passengerData) {
  passengers.push(passengerData);
  savePassengers();
  updatePassengerDisplay();
}
```

---

## Key Features in Detail

### Search & Filter

Users can search trains by:

- Train number (e.g., "91" for Silver Star)
- Route name (e.g., "Northeast Regional")
- Origin/destination stations

**Implementation**:

```javascript
function searchTrains(query) {
  const results = trains.filter((train) => {
    return (
      train.number.includes(query) ||
      train.route.toLowerCase().includes(query.toLowerCase()) ||
      train.origin.toLowerCase().includes(query.toLowerCase()) ||
      train.destination.toLowerCase().includes(query.toLowerCase())
    );
  });

  displaySearchResults(results);
}

// Debounce search to avoid excessive API calls
const debouncedSearch = debounce(searchTrains, 300);

document.getElementById('search-input').addEventListener('input', (e) => {
  debouncedSearch(e.target.value);
});
```

### Booking Workflow

Multi-step booking process:

1. Select train
2. Choose origin/destination stations
3. Enter passenger details
4. Select seat (if available from API)
5. Confirm booking

**Form Validation**:

```javascript
function validateBooking(formData) {
  const errors = [];

  if (!formData.passengerName || formData.passengerName.length < 2) {
    errors.push('Name must be at least 2 characters');
  }

  if (!formData.trainNumber) {
    errors.push('Please select a train');
  }

  if (new Date(formData.departureDate) < new Date()) {
    errors.push('Departure date must be in the future');
  }

  if (errors.length > 0) {
    displayValidationErrors(errors);
    return false;
  }

  return true;
}
```

---

## Responsive Design

The interface adapts to different screen sizes:

**Mobile-First CSS**:

```css
/* Base styles for mobile */
.train-card {
  width: 100%;
  padding: 1rem;
  margin-bottom: 1rem;
}

/* Tablet and up */
@media (min-width: 768px) {
  .train-card {
    width: calc(50% - 1rem);
    display: inline-block;
  }
}

/* Desktop */
@media (min-width: 1024px) {
  .train-card {
    width: calc(33.333% - 1rem);
  }
}
```

**Flexbox for Layouts**:

```css
.train-list {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: space-between;
}

.station-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}
```

---

## Challenges & Solutions

### Challenge 1: Async Data Loading

**Problem**: Multiple API calls needed for complete train information (status + route + stations). UI felt slow.

**Solution**: Implemented **Promise.all()** to parallelize requests:

```javascript
async function loadTrainDetails(trainNumber) {
  showLoadingSpinner();

  try {
    const [status, route, stations] = await Promise.all([
      getTrainStatus(trainNumber),
      getTrainRoute(trainNumber),
      getTrainStations(trainNumber),
    ]);

    displayTrainInfo({ status, route, stations });
  } catch (error) {
    handleError(error);
  } finally {
    hideLoadingSpinner();
  }
}
```

### Challenge 2: API Data Inconsistencies

**Problem**: Amtrak API sometimes returns incomplete data (missing coordinates, null delays).

**Solution**: Defensive programming with fallbacks:

```javascript
function safeGet(obj, path, defaultValue = 'N/A') {
  return (
    path.split('.').reduce((current, prop) => current?.[prop], obj) ??
    defaultValue
  );
}

// Usage
const delay = safeGet(trainData, 'stations.0.delay', 0);
const coordinates = safeGet(trainData, 'coordinates.lat', null);
```

### Challenge 3: State Management

**Problem**: Keeping UI in sync with data as users add/remove passengers.

**Solution**: Simple observer pattern:

```javascript
const stateManager = {
  passengers: [],
  observers: [],

  subscribe(callback) {
    this.observers.push(callback);
  },

  notify() {
    this.observers.forEach((callback) => callback(this.passengers));
  },

  updatePassengers(newPassengers) {
    this.passengers = newPassengers;
    this.notify();
  },
};

// UI subscribes to changes
stateManager.subscribe((passengers) => {
  renderPassengerList(passengers);
});
```

---

## What I Learned

### Technical Skills

**JavaScript Proficiency**:

- Async/await for API calls
- DOM manipulation without jQuery
- ES6+ features (arrow functions, destructuring, template literals)
- Event delegation for dynamic content

**Web APIs**:

- Fetch API for HTTP requests
- localStorage for client-side persistence
- Promises for async operations

**CSS**:

- Flexbox and Grid layouts
- Media queries for responsive design
- CSS variables for theming

### Software Engineering Practices

- **Separation of concerns**: API layer, business logic, and UI rendering are distinct
- **Error handling**: Every API call wrapped in try-catch
- **Code organization**: Functions are small, focused, and reusable
- **Documentation**: Comments explain "why", not "what"

---

## Suggested Improvements for Future Portfolio Enhancement

While the current implementation is functional, here are ideas to make it more visually impressive:

### Visual Enhancements

1. **Interactive Map**: Use Leaflet.js or Mapbox to show train positions on a live map

   ```html
   <div id="map" style="height: 400px;"></div>
   <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
   ```

2. **Animated Route Progress**: Show train moving between stations with CSS animations

3. **Dark Mode**: Toggle between light/dark themes

   ```css
   :root {
     --bg-color: #ffffff;
     --text-color: #000000;
   }

   [data-theme='dark'] {
     --bg-color: #1a1a1a;
     --text-color: #ffffff;
   }
   ```

4. **Charts/Graphs**: Visualize delay statistics with Chart.js
   ```javascript
   new Chart(ctx, {
     type: 'bar',
     data: {
       labels: stationNames,
       datasets: [
         {
           label: 'Delays (minutes)',
           data: delays,
         },
       ],
     },
   });
   ```

### Feature Enhancements

- **Notifications**: Alert users when their train is delayed
- **Price Estimation**: Calculate ticket costs based on distance
- **Multi-leg Trips**: Plan journeys requiring train transfers
- **Historical Data**: Track on-time performance over time

---

## Media Placeholders

_The following media would significantly enhance this project page:_

**Screenshots Needed**:

- `app-interface.png`: Main dashboard with train list
- `booking-form.png`: Passenger booking interface
- `live-tracking.png`: Real-time train status display
- `route-view.png`: Station-by-station route visualization

**Video Demonstration**:

- Screen recording showing: search → select train → view route → add passenger → display booking

**Diagram**:

- System architecture diagram (already described above, needs visual)
- Data flow diagram showing API → Processing → UI

---

## Technologies Used

**Frontend**: HTML5, CSS3 (Flexbox, Grid), Vanilla JavaScript (ES6+)  
**API**: Amtrak API v3 (REST)  
**Storage**: localStorage API  
**Development Tools**: VS Code, Chrome DevTools, Git  
**Testing**: Manual browser testing across Chrome, Firefox, Safari

---

_This project demonstrates practical full-stack web development skills: API integration, state management, and responsive design—foundational skills applicable to any modern web application._
