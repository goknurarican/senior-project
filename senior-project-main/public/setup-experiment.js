// setup-experiment.js
const sqlite3 = require('sqlite3').verbose();
const { open } = require('sqlite');

async function setup() {
  const db = await open({
    filename: './experiment.db',
    driver: sqlite3.Database
  });

  console.log('🔧 Setting up experiment database...\n');

  // Ensure tables exist with all required columns
  await db.exec(`
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      experiment_group TEXT,
      user_agent TEXT,
      ip TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS scenarios (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      type TEXT,
      target_page TEXT,
      selector TEXT,
      params TEXT,
      probability REAL DEFAULT 0.5,
      enabled INTEGER DEFAULT 1,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      event_type TEXT,
      event_data TEXT,
      page_url TEXT,
      timestamp INTEGER,
      relative_t_ms INTEGER,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(session_id) REFERENCES sessions(id)
    );

    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT UNIQUE,
      password TEXT,
      role TEXT DEFAULT 'user',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS products (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      description TEXT,
      price REAL,
      image TEXT,
      category TEXT,
      stock INTEGER DEFAULT 100,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS experiments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      description TEXT,
      start_date TEXT,
      end_date TEXT,
      status TEXT DEFAULT 'active',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  `);

  // Add an admin user
  await db.run(`
    INSERT OR IGNORE INTO users (email, password, role) 
    VALUES ('admin@test.com', 'admin123', 'admin')
  `);

  // Check if scenarios already exist
  const scenarioCount = await db.get('SELECT COUNT(*) as count FROM scenarios');

  if (scenarioCount.count === 0) {
    console.log('📦 Adding default scenarios...');

    const scenarios = [
      // Loading/Visual scenarios
      { name: 'Slow Image Load', type: 'slow_image', target_page: '/products', selector: '.product-image', params: { delay: 2000 }, probability: 0.6 },
      { name: 'Broken Image', type: 'broken_image', target_page: '/products', selector: '.product-image', params: {}, probability: 0.3 },
      { name: 'Skeleton Prolong', type: 'skeleton_prolong', target_page: '/products', selector: '.product-card', params: { delay: 3000 }, probability: 0.6 },

      // Interaction/Friction scenarios
      { name: 'Button Delay', type: 'button_delay', target_page: '*', selector: 'button', params: { delay: 1500 }, probability: 0.6 },
      { name: 'First Click Miss', type: 'first_click_miss', target_page: '*', selector: 'button', params: {}, probability: 0.3 },
      { name: 'Feedback Late', type: 'feedback_late', target_page: '*', selector: '', params: { delay: 2000 }, probability: 0.6 },

      // Search/Navigation scenarios
      { name: 'Search Irrelevant', type: 'search_irrelevant', target_page: '/products', selector: '', params: { duration: 5000 }, probability: 0.6 },
      { name: 'Facet Reset Once', type: 'facet_reset_once', target_page: '/products', selector: '', params: {}, probability: 0.3 },
      { name: 'Sort Reset', type: 'sort_reset', target_page: '/products', selector: '', params: {}, probability: 0.3 },

      // Cart/Checkout scenarios
      { name: 'Price Change Warning', type: 'price_change', target_page: '/checkout', selector: '', params: { change_percent: 5 }, probability: 0.6 },
      { name: 'Coupon Min Spend', type: 'coupon_min_spend', target_page: '/cart', selector: '', params: {}, probability: 0.6 },
      { name: 'Coupon Expired', type: 'coupon_expired', target_page: '/cart', selector: '', params: {}, probability: 0.3 },

      // Payment scenarios
      { name: '3DS Soft Fail', type: '3ds_soft_fail', target_page: '/checkout', selector: '', params: {}, probability: 0.6 },
      { name: 'Payment Retry Timeout', type: 'payment_retry_timeout', target_page: '/checkout', selector: '', params: {}, probability: 0.6 },

      // Overlay scenarios
      { name: 'Overlay Blocking', type: 'overlay_blocking', target_page: '/', selector: '', params: { duration: 4000 }, probability: 0.3 },

      // Network scenarios
      { name: 'Network Jitter', type: 'network_jitter', target_page: '*', selector: '', params: { delay: 500 }, probability: 0.6 }
    ];

    for (const scenario of scenarios) {
      await db.run(
        'INSERT INTO scenarios (name, type, target_page, selector, params, probability, enabled) VALUES (?, ?, ?, ?, ?, ?, ?)',
        [scenario.name, scenario.type, scenario.target_page, scenario.selector, JSON.stringify(scenario.params), scenario.probability, 1]
      );
    }

    console.log(`✅ Added ${scenarios.length} scenarios`);
  } else {
    console.log(`✓ Scenarios already exist (${scenarioCount.count} scenarios)`);
  }

  // Check if products exist
  const productCount = await db.get('SELECT COUNT(*) as count FROM products');

  if (productCount.count === 0) {
    console.log('🛍️ Adding sample products...');

    const products = [
      { name: 'Laptop Pro', description: 'High-performance laptop', price: 1299.99, image: '/images/laptop.png', category: 'electronics' },
      { name: 'Wireless Mouse', description: 'Ergonomic wireless mouse', price: 29.99, image: '/images/wireless-mouse.jpg', category: 'electronics' },
      { name: 'Mechanical Keyboard', description: 'RGB mechanical keyboard', price: 149.99, image: '/images/MechanicalKeyboard.jpg', category: 'electronics' },
      { name: 'Monitor 27"', description: '4K HDR monitor', price: 499.99, image: '/images/Monitor.jpg', category: 'electronics' },
      { name: 'USB Hub', description: '7-port USB 3.0 hub', price: 39.99, image: '/images/hub.jpg', category: 'electronics' },
      { name: 'Webcam HD', description: '1080p webcam with mic', price: 79.99, image: '/images/WebcamHD.jpg', category: 'electronics' },
      { name: 'Standing Desk', description: 'Adjustable standing desk', price: 599.99, image: '/images/StandingDesk.jpg', category: 'home' },
      { name: 'Office Chair', description: 'Ergonomic office chair', price: 299.99, image: '/images/OfficeChair.jpg', category: 'home' },
      { name: 'Desk Lamp LED', description: 'Smart LED desk lamp', price: 49.99, image: '/images/DeskLampLED.jpg', category: 'home' },
      { name: 'Bookshelf', description: '5-tier bookshelf', price: 149.99, image: '/images/Bookshelf.jpeg', category: 'home' }
    ];

    for (const product of products) {
      await db.run(
        'INSERT INTO products (name, description, price, image, category) VALUES (?, ?, ?, ?, ?)',
        [product.name, product.description, product.price, product.image, product.category]
      );
    }

    console.log(`✅ Added ${products.length} products`);
  } else {
    console.log(`✓ Products already exist (${productCount.count} products)`);
  }

  // Create an initial experiment
  const experimentCount = await db.get('SELECT COUNT(*) as count FROM experiments');

  if (experimentCount.count === 0) {
    await db.run(`
      INSERT INTO experiments (name, description, start_date, end_date, status)
      VALUES ('Main Experiment', 'Testing negative UX patterns impact', datetime('now'), datetime('now', '+30 days'), 'active')
    `);
    console.log('✅ Created main experiment');
  }

  // Show current stats
  console.log('\n📊 Current Database Stats:');
  console.log(`   Sessions: ${(await db.get('SELECT COUNT(*) as count FROM sessions')).count}`);
  console.log(`   Scenarios: ${scenarioCount.count > 0 ? scenarioCount.count : (await db.get('SELECT COUNT(*) as count FROM scenarios')).count} (${(await db.get('SELECT COUNT(*) as count FROM scenarios WHERE enabled = 1')).count} enabled)`);
  console.log(`   Events: ${(await db.get('SELECT COUNT(*) as count FROM events')).count}`);
  console.log(`   Products: ${(await db.get('SELECT COUNT(*) as count FROM products')).count}`);

  console.log('\n✨ Setup complete!');
  console.log('   Admin login: admin@test.com / admin123');
  console.log('   Start the dev server: npm run dev');

  await db.close();
}

setup().catch(console.error);