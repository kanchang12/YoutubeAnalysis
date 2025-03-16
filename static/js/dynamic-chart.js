// dynamic-chart.js - No React or Node.js required
function createDynamicChart(containerId, data, chartType, title) {
  // Clear the container
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  
  // Set chart title
  const titleElement = document.createElement('h3');
  titleElement.textContent = title || 'YouTube Data Analysis';
  titleElement.className = 'chart-title';
  container.appendChild(titleElement);
  
  // Add chart type selector
  const selectorDiv = document.createElement('div');
  selectorDiv.className = 'chart-type-selector';
  
  const types = ['pie', 'bar', 'line'];
  types.forEach(type => {
    const button = document.createElement('button');
    button.textContent = type.charAt(0).toUpperCase() + type.slice(1);
    button.className = `chart-btn ${type === chartType ? 'active' : ''}`;
    button.onclick = function() {
      createDynamicChart(containerId, data, type, title);
    };
    selectorDiv.appendChild(button);
  });
  
  container.appendChild(selectorDiv);
  
  // Create canvas for chart
  const canvas = document.createElement('canvas');
  canvas.id = 'chart-canvas';
  container.appendChild(canvas);
  
  // Create chart using Chart.js
  const ctx = canvas.getContext('2d');
  
  // Prepare data
  const labels = data.map(item => item.name);
  const values = data.map(item => item.value);
  const colors = generateColors(data.length);
  
  // Create appropriate chart type
  if (chartType === 'pie') {
    new Chart(ctx, {
      type: 'pie',
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: colors,
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'right',
          },
          title: {
            display: false
          }
        }
      }
    });
  } else if (chartType === 'bar') {
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Value',
          data: values,
          backgroundColor: colors[0],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
  } else if (chartType === 'line') {
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Value',
          data: values,
          borderColor: colors[0],
          tension: 0.1,
          fill: false
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
  }
}

// Helper function to generate colors
function generateColors(count) {
  const baseColors = [
    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
    '#FF9F40', '#8AC148', '#F77825', '#00A8C6', '#D4E09B'
  ];
  
  if (count <= baseColors.length) {
    return baseColors.slice(0, count);
  }
  
  // If we need more colors, generate them
  const colors = [...baseColors];
  for (let i = baseColors.length; i < count; i++) {
    const r = Math.floor(Math.random() * 255);
    const g = Math.floor(Math.random() * 255);
    const b = Math.floor(Math.random() * 255);
    colors.push(`rgb(${r}, ${g}, ${b})`);
  }
  
  return colors;
}
