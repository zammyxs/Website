// Global state management
const state = {
    currentPage: 'login',
    isLoggedIn: false,
    inventory: JSON.parse(localStorage.getItem('inventory') || '{}'),
  };
  
  // DOM Elements
  const pages = {
    login: document.getElementById('loginPage'),
    dashboard: document.getElementById('dashboardPage'),
    inventory: document.getElementById('inventoryPage'),
    manage: document.getElementById('managePage'),
    notFound: document.getElementById('notFoundPage'),
  };
  
  const appLayout = document.getElementById('appLayout');
  const navbar = document.getElementById('navbar');
  
  // Navigation
  function navigateTo(page) {
    if (!state.isLoggedIn && page !== 'login') {
      navigateTo('login');
      return;
    }
  
    // Hide all pages
    Object.values(pages).forEach(p => p.classList.remove('active'));
    
    // Show selected page
    if (pages[page]) {
      pages[page].classList.add('active');
      state.currentPage = page;
      
      // Toggle app layout visibility
      if (page === 'login') {
        appLayout.style.display = 'none';
      } else {
        appLayout.style.display = 'block';
        updateNavLinks();
        updatePageData();
      }
    } else {
      pages.notFound.classList.add('active');
    }
  }
  
  // Update navigation links active state
  function updateNavLinks() {
    document.querySelectorAll('.nav-link').forEach(link => {
      const page = link.dataset.page;
      if (page === state.currentPage) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  }
  
  // Update page data based on current inventory
  function updatePageData() {
    const inventory = state.inventory;
    
    // Dashboard updates
    if (state.currentPage === 'dashboard') {
      const totalProducts = Object.keys(inventory).length;
      const totalItems = Object.values(inventory).reduce((sum, item) => sum + item.quantity, 0);
      const totalValue = Object.entries(inventory).reduce((sum, [_, item]) => sum + (item.quantity * item.price), 0);
      
      document.getElementById('totalProducts').textContent = totalProducts;
      document.getElementById('totalItems').textContent = totalItems;
      document.getElementById('totalValue').textContent = `₱${totalValue.toFixed(2)}`;
      
      // Update product table
      const tableBody = document.getElementById('productTableBody');
      const noProductsMessage = document.getElementById('noProductsMessage');
      
      if (totalProducts > 0) {
        tableBody.innerHTML = '';
        Object.entries(inventory)
          .sort(([, a], [, b]) => (b.quantity * b.price) - (a.quantity * a.price))
          .slice(0, 5)
          .forEach(([name, item]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
              <td>${name}</td>
              <td>${item.quantity}</td>
              <td>₱${item.price.toFixed(2)}</td>
              <td>₱${(item.quantity * item.price).toFixed(2)}</td>
            `;
            tableBody.appendChild(row);
          });
        noProductsMessage.style.display = 'none';
      } else {
        noProductsMessage.style.display = 'block';
      }
    }
    
    // Management page updates
    if (state.currentPage === 'manage') {
      const tableBody = document.getElementById('manageTableBody');
      const manageTableContainer = document.getElementById('manageTableContainer');
      const noItemsMessage = document.getElementById('noItemsMessage');
      const totalValue = Object.entries(inventory).reduce((sum, [_, item]) => sum + (item.quantity * item.price), 0);
      
      document.getElementById('totalInventoryValue').querySelector('span').textContent = `₱${totalValue.toFixed(2)}`;
      
      if (Object.keys(inventory).length > 0) {
        tableBody.innerHTML = '';
        Object.entries(inventory).forEach(([name, item]) => {
          const row = document.createElement('tr');
          row.innerHTML = `
            <td>
              <div class="editable-field" onclick="toggleEdit(this, '${name}', 'name')">
                ${name}
              </div>
            </td>
            <td>
              <div class="quantity-controls">
                <button class="btn btn-ghost" onclick="updateQuantity('${name}', -1)">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>
                </button>
                <span>${item.quantity}</span>
                <button class="btn btn-ghost" onclick="updateQuantity('${name}', 1)">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                </button>
              </div>
            </td>
            <td>
              <div class="editable-field" onclick="toggleEdit(this, '${name}', 'price')">
                ${item.price.toFixed(2)}
              </div>
            </td>
            <td>₱${(item.quantity * item.price).toFixed(2)}</td>
            <td>
              <button class="btn btn-danger" onclick="removeItem('${name}')">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
              </button>
            </td>
          `;
          tableBody.appendChild(row);
        });
        manageTableContainer.style.display = 'block';
        noItemsMessage.style.display = 'none';
      } else {
        manageTableContainer.style.display = 'none';
        noItemsMessage.style.display = 'block';
      }
    }
  }
  
  // Event Handlers
  document.getElementById('loginForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    if (username === 'jekzstore' && password === 'password123') {
      state.isLoggedIn = true;
      showToast('success', 'Login successful!');
      navigateTo('dashboard');
    } else {
      showToast('error', 'Invalid username or password');
    }
  });
  
  document.getElementById('inventoryForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const itemName = document.getElementById('itemName').value.trim();
    const quantity = parseInt(document.getElementById('quantity').value);
    const price = parseFloat(document.getElementById('price').value);
    
    if (!itemName || isNaN(quantity) || isNaN(price) || quantity <= 0 || price <= 0) {
      showToast('error', 'Please fill all fields correctly');
      return;
    }
    
    const inventory = state.inventory;
    if (inventory[itemName]) {
      inventory[itemName].quantity += quantity;
    } else {
      inventory[itemName] = { quantity, price };
    }
    
    state.inventory = inventory;
    localStorage.setItem('inventory', JSON.stringify(inventory));
    showToast('success', `${itemName} added to inventory!`);
    
    // Reset form
    e.target.reset();
    updatePageData();
  });
  
  document.getElementById('searchInput')?.addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase();
    const rows = document.getElementById('manageTableBody').getElementsByTagName('tr');
    
    Array.from(rows).forEach(row => {
      const itemName = row.cells[0].textContent.toLowerCase();
      row.style.display = itemName.includes(searchTerm) ? '' : 'none';
    });
  });
  
  // Navigation event listeners
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const page = link.dataset.page;
      if (page) navigateTo(page);
    });
  });
  
  document.getElementById('logoutBtn').addEventListener('click', (e) => {
    e.preventDefault();
    state.isLoggedIn = false;
    navigateTo('login');
    showToast('success', 'Logged out successfully');
  });
  
  document.getElementById('addFirstItemBtn')?.addEventListener('click', () => {
    navigateTo('inventory');
  });
  
  document.getElementById('backToHomeBtn')?.addEventListener('click', () => {
    navigateTo('dashboard');
  });
  
  // Inventory Management Functions
  function updateQuantity(itemName, change) {
    const inventory = state.inventory;
    const newQuantity = inventory[itemName].quantity + change;
    
    if (newQuantity >= 0) {
      inventory[itemName].quantity = newQuantity;
      state.inventory = inventory;
      localStorage.setItem('inventory', JSON.stringify(inventory));
      updatePageData();
      showToast('success', `Updated ${itemName} quantity`);
    }
  }
  
  function removeItem(itemName) {
    const inventory = state.inventory;
    delete inventory[itemName];
    state.inventory = inventory;
    localStorage.setItem('inventory', JSON.stringify(inventory));
    updatePageData();
    showToast('success', `Removed ${itemName} from inventory`);
  }
  
  function toggleEdit(element, itemName, field) {
    const currentValue = field === 'name' ? itemName : state.inventory[itemName].price;
    const input = document.createElement('input');
    input.type = field === 'price' ? 'number' : 'text';
    input.value = currentValue;
    input.step = field === 'price' ? '0.01' : undefined;
    input.min = field === 'price' ? '0.01' : undefined;
    input.className = 'editable-input';
    
    const save = () => {
      const newValue = input.value.trim();
      if (field === 'name' && newValue && newValue !== itemName) {
        const inventory = state.inventory;
        inventory[newValue] = inventory[itemName];
        delete inventory[itemName];
        state.inventory = inventory;
        localStorage.setItem('inventory', JSON.stringify(inventory));
        updatePageData();
        showToast('success', `Renamed ${itemName} to ${newValue}`);
      } else if (field === 'price') {
        const newPrice = parseFloat(newValue);
        if (!isNaN(newPrice) && newPrice > 0) {
          const inventory = state.inventory;
          inventory[itemName].price = newPrice;
          state.inventory = inventory;
          localStorage.setItem('inventory', JSON.stringify(inventory));
          updatePageData();
          showToast('success', `Updated ${itemName} price`);
        }
      }
    };
    
    input.addEventListener('blur', save);
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        save();
      }
    });
    
    element.innerHTML = '';
    element.appendChild(input);
    input.focus();
  }
  
  // Toast Notification System
  function showToast(type, message) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <div class="toast-icon">
        ${type === 'success' 
          ? '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
          : '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
        }
      </div>
      <div class="toast-content">
        <div class="toast-title">${type === 'success' ? 'Success' : 'Error'}</div>
        <div class="toast-message">${message}</div>
      </div>
    `;
    
    document.getElementById('toastContainer').appendChild(toast);
    
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }
  
  // Navbar scroll effect
  window.addEventListener('scroll', () => {
    if (window.scrollY > 10) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });
  
  // Update copyright year
  const currentYear = new Date().getFullYear();
  document.getElementById('currentYear').textContent = currentYear;
  document.getElementById('footerYear').textContent = currentYear;
  
  // Initial page load
  navigateTo('login');

document.addEventListener('DOMContentLoaded', () => {
  const previewPage = document.getElementById('previewPage');
  const loginPage = document.getElementById('loginPage');
  const appLayout = document.getElementById('appLayout');
  const goToLogin = document.getElementById('goToLogin');
  const logoutBtn = document.getElementById('logoutBtn');
  const productTableBody = document.getElementById('productTableBody');
  const noProductsMessage = document.getElementById('noProductsMessage');

  // Sample data for available items
  const products = [
    { name: 'Product 1', stock: 10, price: 100, totalValue: 1000 },
    { name: 'Product 2', stock: 5, price: 200, totalValue: 1000 },
    { name: 'Product 3', stock: 20, price: 50, totalValue: 1000 },
  ];

  // Populate the product table
  if (products.length > 0) {
    noProductsMessage.style.display = 'none';
    products.forEach(product => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${product.name}</td>
        <td>${product.stock}</td>
        <td>${product.price}</td>
        <td>${product.totalValue}</td>
      `;
      productTableBody.appendChild(row);
    });
  } else {
    noProductsMessage.style.display = 'block';
  }

  // Handle navigation to login page
  goToLogin.addEventListener('click', (e) => {
    e.preventDefault();
    previewPage.classList.remove('active');
    loginPage.classList.add('active');
  });

  // Handle logout functionality
  logoutBtn.addEventListener('click', (e) => {
    e.preventDefault();
    appLayout.classList.remove('active');
    previewPage.classList.add('active');
  });

  document.getElementById('logoutBtn').addEventListener('click', function() {
    // Show the logout options modal
    document.getElementById('logoutModal').style.display = 'block';

    // Set the current date
    const currentDate = new Date().toLocaleDateString();
    document.getElementById('logoutDate').textContent = currentDate;

    // Set the number of products for today (example value, replace with actual logic)
    const productsToday = 10; // Replace with actual logic to get the number of products
    document.getElementById('productsToday').textContent = productsToday;
  });
  
  document.getElementById('viewPreviewBtn').addEventListener('click', function() {
    // Hide the modal
    document.getElementById('logoutModal').style.display = 'none';
    // Show the preview page and hide other pages
    document.getElementById('previewPage').classList.add('active');
    document.getElementById('loginPage').classList.remove('active');
    document.getElementById('appLayout').classList.remove('active');
  });
  
  document.getElementById('loginAgainBtn').addEventListener('click', function() {
    // Hide the modal
    document.getElementById('logoutModal').style.display = 'none';
    // Show the login page and hide other pages
    document.getElementById('loginPage').classList.add('active');
    document.getElementById('previewPage').classList.remove('active');
    document.getElementById('appLayout').classList.remove('active');
  });
});

document.addEventListener('DOMContentLoaded', function() {
  // Set the current year in the footer
  document.getElementById('footerYear').textContent = new Date().getFullYear();

  // Set the current date in the logout modal
  const currentDate = new Date().toLocaleDateString();
  document.getElementById('logoutDate').textContent = currentDate;

  // Set the number of products for the day (example value, replace with actual logic)
  const productsToday = 5; // Replace with actual logic to get the number of products
  document.getElementById('productsToday').textContent = productsToday;
});
fetch('/inventory')  // Ensure this endpoint is the same for both pages
    .then(response => response.json())
    .then(data => {
        updateProductTable(data);  // Use the same function for both pages
    })
    .catch(error => console.error('Error fetching inventory:', error));
    function exportToExcel() {
      const inventory = state.inventory;
      if (Object.keys(inventory).length === 0) {
          alert("No inventory data to export.");
          return;
      }
  
      let csvContent = "data:text/csv;charset=utf-8,";
      csvContent += "Product,Stock,Price,Total Value\n"; 
  
      Object.entries(inventory).forEach(([name, item]) => {
          let row = `${name},${item.quantity},${item.price},${(item.quantity * item.price).toFixed(2)}`;
          csvContent += row + "\n";
      });
  
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "inventory_data.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
  }
  function exportToExcel() {
    const inventory = state.inventory;

    if (Object.keys(inventory).length === 0) {
        alert("No inventory data to export.");
        return;
    }

    // Convert inventory object to an array of arrays for Excel
    let data = [["Product", "Stock", "Price", "Total Value"]];

    Object.entries(inventory).forEach(([name, item]) => {
        data.push([name, item.quantity, item.price, (item.quantity * item.price).toFixed(2)]);
    });

    // Create a new worksheet
    let worksheet = XLSX.utils.aoa_to_sheet(data);

    // Create a new workbook and append the worksheet
    let workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Inventory");

    // Save the file
    XLSX.writeFile(workbook, "inventory_data.xlsx");
}

