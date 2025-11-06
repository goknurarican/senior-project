// lib/db.ts
import { open } from 'sqlite';
import sqlite3 from 'sqlite3';
import path from 'path';

let db: any = null;

export async function getDb() {
  if (!db) {
    db = await open({
      filename: path.join(process.cwd(), 'experiment.db'),
      driver: sqlite3.Database
    });
    
    await initDb();
  }
  return db;
}

async function initDb() {
  await db.exec(`
    -- Sessions table
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      experiment_group TEXT DEFAULT 'control',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      user_agent TEXT,
      ip TEXT
    );

    -- Events table
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      event_type TEXT,
      event_data TEXT,
      page_url TEXT,
      timestamp INTEGER,
      relative_t_ms INTEGER,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    -- Experiments table
    CREATE TABLE IF NOT EXISTS experiments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      status TEXT DEFAULT 'draft',
      control_weight INTEGER DEFAULT 50,
      variant_a_weight INTEGER DEFAULT 25,
      variant_b_weight INTEGER DEFAULT 25,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Scenarios table
    CREATE TABLE IF NOT EXISTS scenarios (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      type TEXT,
      target_page TEXT,
      selector TEXT,
      params TEXT,
      probability REAL DEFAULT 0.1,
      enabled INTEGER DEFAULT 1,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Products table
    CREATE TABLE IF NOT EXISTS products (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT,
      price REAL,
      image TEXT,
      category TEXT,
      stock INTEGER DEFAULT 100,
      description TEXT
    );

    -- Cart items
    CREATE TABLE IF NOT EXISTS cart_items (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      product_id INTEGER,
      quantity INTEGER DEFAULT 1,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (session_id) REFERENCES sessions(id),
      FOREIGN KEY (product_id) REFERENCES products(id)
    );

    -- Scenario triggers log
    CREATE TABLE IF NOT EXISTS scenario_triggers (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      scenario_id INTEGER,
      status TEXT,
      triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (session_id) REFERENCES sessions(id),
      FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
    );

    -- Users table for authentication
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT UNIQUE,
      password TEXT,
      role TEXT DEFAULT 'user',
      name TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  `);

  // Seed products if empty
  const count = await db.get('SELECT COUNT(*) as count FROM products');
  if (count.count === 0) {
    await seedProducts();
  }

  // Seed scenarios if empty - CLEAR OLD AND RECREATE
  const scenarioCount = await db.get('SELECT COUNT(*) as count FROM scenarios');
  if (scenarioCount.count > 0) {
    await db.run('DELETE FROM scenarios');
  }
  await seedScenarios();

  // Seed users if empty
  const userCount = await db.get('SELECT COUNT(*) as count FROM users');
  if (userCount.count === 0) {
    await seedUsers();
  }
}

async function seedProducts() {
  const products = [
    { title: 'Laptop Pro X1', price: 15999.99, category: 'electronics', image: 'https://picsum.photos/400/300?random=1' },
    { title: 'Wireless Mouse', price: 299.99, category: 'electronics', image: 'https://picsum.photos/400/300?random=2' },
    { title: 'USB-C Hub', price: 549.99, category: 'electronics', image: 'https://picsum.photos/400/300?random=3' },
    { title: 'Mechanical Keyboard', price: 1299.99, category: 'electronics', image: 'https://picsum.photos/400/300?random=4' },
    { title: 'Monitor 27"', price: 7999.99, category: 'electronics', image: 'https://picsum.photos/400/300?random=5' },
    { title: 'Webcam HD', price: 899.99, category: 'electronics', image: 'https://picsum.photos/400/300?random=6' },
    { title: 'Desk Lamp LED', price: 449.99, category: 'home', image: 'https://picsum.photos/400/300?random=7' },
    { title: 'Office Chair', price: 3999.99, category: 'home', image: 'https://picsum.photos/400/300?random=8' },
    { title: 'Standing Desk', price: 8999.99, category: 'home', image: 'https://picsum.photos/400/300?random=9' },
    { title: 'Coffee Maker', price: 1899.99, category: 'home', image: 'https://picsum.photos/400/300?random=10' },
  ];

  for (const p of products) {
    await db.run(
      'INSERT INTO products (title, price, category, image, description) VALUES (?, ?, ?, ?, ?)',
      [p.title, p.price, p.category, p.image, `High quality ${p.title} with premium features and warranty.`]
    );
  }
}

async function seedScenarios() {
  const scenarios = [
    // A) Loading/Visual Scenarios
    {
      name: 'Slow Image Load',
      type: 'slow_image',
      target_page: '/products',
      selector: '.product-image',
      params: JSON.stringify({ delay: 1500 }),
      probability: 0.6
    },
    {
      name: 'Broken Image',
      type: 'broken_image',
      target_page: '/products',
      selector: '.product-image',
      params: JSON.stringify({ probability: 0.05 }),
      probability: 0.3
    },
    {
      name: 'Skeleton Prolong',
      type: 'skeleton_prolong',
      target_page: '/products',
      selector: '.product-card',
      params: JSON.stringify({ delay: 2000 }),
      probability: 0.6
    },
    
    // B) Interaction/Friction Scenarios
    {
      name: 'Button Delay',
      type: 'button_delay',
      target_page: '*',
      selector: '.add-to-cart',
      params: JSON.stringify({ delay: 1200 }),
      probability: 0.6
    },
    {
      name: 'First Click Miss',
      type: 'first_click_miss', 
      target_page: '*',
      selector: 'button',
      params: JSON.stringify({}),
      probability: 0.3
    },
    {
      name: 'Feedback Late',
      type: 'feedback_late',
      target_page: '*',
      selector: null,
      params: JSON.stringify({ delay: 1500 }),
      probability: 0.6
    },
    
    // C) Search/Navigation Scenarios
    {
      name: 'Search Irrelevant',
      type: 'search_irrelevant',
      target_page: '/products',
      selector: null,
      params: JSON.stringify({ duration: 5000 }),
      probability: 0.6
    },
    {
      name: 'Facet Reset Once',
      type: 'facet_reset_once',
      target_page: '/products',
      selector: null,
      params: JSON.stringify({}),
      probability: 0.3
    },
    {
      name: 'Sort Reset',
      type: 'sort_reset',
      target_page: '/products',
      selector: null,
      params: JSON.stringify({}),
      probability: 0.3
    },
    
    // D) Data Consistency Scenarios
    {
      name: 'Price Change Warning',
      type: 'price_change',
      target_page: '/checkout',
      selector: null,
      params: JSON.stringify({ change_percent: 5 }),
      probability: 0.6
    },
    
    // E) Cart/Coupon Scenarios
    {
      name: 'Coupon Min Spend',
      type: 'coupon_min_spend',
      target_page: '/cart',
      selector: null,
      params: JSON.stringify({ min_amount: 500 }),
      probability: 0.6
    },
    {
      name: 'Coupon Expired',
      type: 'coupon_expired',
      target_page: '/cart',
      selector: null,
      params: JSON.stringify({}),
      probability: 0.3
    },
    
    // F) Payment Scenarios
    {
      name: '3DS Soft Fail',
      type: '3ds_soft_fail',
      target_page: '/checkout',
      selector: null,
      params: JSON.stringify({}),
      probability: 0.6
    },
    {
      name: 'Payment Retry Timeout',
      type: 'payment_retry_timeout',
      target_page: '/checkout',
      selector: null,
      params: JSON.stringify({ timeout: 1500 }),
      probability: 0.6
    },
    
    // G) Overlay/Attention Scenarios  
    {
      name: 'Overlay Blocking',
      type: 'overlay_blocking',
      target_page: '/',
      selector: null,
      params: JSON.stringify({ duration: 4000 }),
      probability: 0.3
    },
    
    // H) Network Scenarios
    {
      name: 'Network Jitter',
      type: 'network_jitter',
      target_page: '*',
      selector: null,
      params: JSON.stringify({ delay: 500 }),
      probability: 0.6
    }
  ];

  for (const s of scenarios) {
    await db.run(
      'INSERT INTO scenarios (name, type, target_page, selector, params, probability, enabled) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [s.name, s.type, s.target_page, s.selector, s.params, s.probability, 1] // All enabled by default
    );
  }
}

async function seedUsers() {
  const users = [
    {
      email: 'admin@test.com',
      password: 'admin123', // In production, use bcrypt
      role: 'admin',
      name: 'Admin User'
    },
    {
      email: 'user@test.com',
      password: 'user123',
      role: 'user',
      name: 'Test User'
    }
  ];

  for (const u of users) {
    await db.run(
      'INSERT INTO users (email, password, role, name) VALUES (?, ?, ?, ?)',
      [u.email, u.password, u.role, u.name]
    );
  }
}

export default getDb;
