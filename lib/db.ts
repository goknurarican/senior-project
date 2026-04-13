// lib/db.ts
import { open } from "sqlite";
import sqlite3 from "sqlite3";
import path from "path";
import bcrypt from "bcryptjs";

// TypeScript için global değişken tanımı
declare global {
  var dbInstance: any;
}

export async function getDb() {
  // Eğer globalde zaten açık bir veritabanı varsa, onu kullan.
  // Bu sayede initDb() tekrar tekrar çalışmaz.
  if (global.dbInstance) {
    return global.dbInstance;
  }

  // Yoksa yeni bağlantı aç
  const db = await open({
    filename: path.join(process.cwd(), "experiment.db"),
    driver: sqlite3.Database,
  });

  // Concurrent access settings — must be set before initDb touches any table
  await db.exec("PRAGMA journal_mode=WAL");    // allow concurrent readers + 1 writer
  await db.exec("PRAGMA busy_timeout=10000"); // wait up to 10s instead of failing instantly
  await db.exec("PRAGMA synchronous=NORMAL"); // faster writes, still safe with WAL

  // Oluşturulan bağlantıyı globale kaydet
  global.dbInstance = db;

  // Veritabanı tablolarını ve verilerini kur
  await initDb(db);

  return db;
}

// db parametresini artık dışarıdan alıyoruz
async function initDb(db: any) {
  await db.exec(`
    -- Sessions table(user_id eklendi)
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      user_id INTEGER,  -- YENİ: Login olan user buraya yazılacak
      experiment_group TEXT DEFAULT 'control',
      phase TEXT DEFAULT 'control',
      assigned_variant TEXT,                  
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      user_agent TEXT,
      ip TEXT
    );

    -- Events table (experiment_group eklendi)
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      user_id INTEGER, -- YENİ: Opsiyonel, analiz kolaylığı için
      experiment_group TEXT, -- YENİ: Event anındaki grup (Control mü C mi?)
      phase TEXT,
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

  // --- ÜRÜN SEED İŞLEMİ (GÜNCELLENDİ) ---
  // "if count > 0" kontrolünü kaldırdık.
  // Her zaman temizlik yapıyoruz ki yarış durumu olsa bile temiz başlasın.

  await db.run("DELETE FROM products");
  await db.run("DELETE FROM cart_items");
  await db.run("DELETE FROM sqlite_sequence WHERE name='products'");

  // Şimdi temizce ekle
  await seedProducts(db);


  // --- SENARYO SEED İŞLEMİ ---
  await db.run("DELETE FROM scenarios");
  await db.run("DELETE FROM sqlite_sequence WHERE name='scenarios'");
  await seedScenarios(db);

  // --- KULLANICI SEED İŞLEMİ ---
  const userCount = await db.get("SELECT COUNT(*) as count FROM users");
  if (userCount.count === 0) {
    await seedUsers(db);
  } else {
    // Migration: fix plain-text passwords from old seed (bcrypt hashes start with $2)
    const fixList = [
      { email: 'admin@test.com', plain: 'admin123' },
      { email: 'user@test.com',  plain: 'user123'  },
    ];
    for (const u of fixList) {
      const row = await db.get('SELECT id, password FROM users WHERE email = ?', [u.email]);
      if (row && !String(row.password).startsWith('$2')) {
        const hashed = await bcrypt.hash(u.plain, 10);
        await db.run('UPDATE users SET password = ? WHERE email = ?', [hashed, u.email]);
      }
    }
  }
}

// Fonksiyonlar artık db nesnesini parametre olarak alıyor
async function seedProducts(db: any) {
  const products = [
    {
      title: "Laptop Pro X1",
      price: 15999.99,
      category: "electronics",
      image: "/images/laptop.png",
    },
    {
      title: "Wireless Mouse",
      price: 299.99,
      category: "electronics",
      image: "/images/wireless-mouse.jpg",
    },
    {
      title: "USB-C Hub",
      price: 549.99,
      category: "electronics",
      image: "/images/hub.jpg",
    },
    {
      title: "Mechanical Keyboard",
      price: 1299.99,
      category: "electronics",
      image: "/images/MechanicalKeyboard.jpg",
    },
    {
      title: 'Monitor 27"',
      price: 7999.99,
      category: "electronics",
      image: "/images/Monitor.jpg",
    },
    {
      title: "Webcam HD",
      price: 899.99,
      category: "electronics",
      image: "/images/WebcamHD.jpg",
    },
    {
      title: "Desk Lamp LED",
      price: 449.99,
      category: "home",
      image: "/images/DeskLampLED.jpg",
    },
    {
      title: "Office Chair",
      price: 3999.99,
      category: "home",
      image: "/images/OfficeChair.jpg",
    },
    {
      title: "Standing Desk",
      price: 8999.99,
      category: "home",
      image: "/images/StandingDesk.jpg",
    },
    {
      title: "Coffee Maker",
      price: 1899.99,
      category: "home",
      image: "/images/CoffeeMaker.jpeg",
    },
    {
      title: "Bluetooth Speaker",
      price: 799.99,
      category: "electronics",
      image: "/images/BluetoothSpeaker.jpg",
    },
    {
      title: "Gaming Headset",
      price: 1499.99,
      category: "electronics",
      image: "/images/GamingHeadset.jpg",
    },
    {
      title: "Smart Watch",
      price: 3499.99,
      category: "electronics",
      image: "/images/SmartWatch.jpg",
    },
    {
      title: "Portable SSD 1TB",
      price: 1999.99,
      category: "electronics",
      image: "/images/PortableSSD1TB.jpg",
    },
    {
      title: "Robot Vacuum",
      price: 6999.99,
      category: "home",
      image: "/images/RobotVacuum.jpg",
    },
    {
      title: "Air Purifier",
      price: 2799.99,
      category: "home",
      image: "/images/AirPurifier.jpg",
    },
    {
      title: "Bookshelf",
      price: 1299.99,
      category: "home",
      image: "/images/Bookshelf.jpeg",
    },
    {
      title: "Smart Lock",
      price: 2499.99,
      category: "home",
      image: "/images/SmartLock.jpg",
    },
    {
      title: "Power Bank 20000mAh",
      price: 899.99,
      category: "electronics",
      image: "/images/PowerBank20000mAh.jpg",
    },
    {
      title: "Digital Photo Frame",
      price: 1199.99,
      category: "home",
      image: "/images/DigitalPhotoFrame.jpeg",
    },
  ];

  for (const p of products) {
    await db.run(
      "INSERT INTO products (title, price, category, image, description) VALUES (?, ?, ?, ?, ?)",
      [
        p.title,
        p.price,
        p.category,
        p.image,
        `High quality ${p.title} with premium features and warranty.`,
      ]
    );
  }
}

async function seedScenarios(db: any) {
  const scenarios = [
    // A) Loading/Visual Scenarios
    {
      name: "Slow Image Load",
      type: "slow_image",
      target_page: "/products",
      selector: ".product-image",
      params: JSON.stringify({ delay: 1500 }),
      probability: 0.6,
    },
    {
      name: "Broken Image",
      type: "broken_image",
      target_page: "/products",
      selector: ".product-image",
      params: JSON.stringify({ probability: 0.05 }),
      probability: 0.3,
    },
    {
      name: "Skeleton Prolong",
      type: "skeleton_prolong",
      target_page: "/products",
      selector: ".product-card",
      params: JSON.stringify({ delay: 2000 }),
      probability: 0.6,
    },

    // B) Interaction/Friction Scenarios
    {
      name: "Button Delay",
      type: "button_delay",
      target_page: "*",
      selector: ".add-to-cart",
      params: JSON.stringify({ delay: 1200 }),
      probability: 0.6,
    },
    {
      name: "First Click Miss",
      type: "first_click_miss",
      target_page: "*",
      selector: "button",
      params: JSON.stringify({}),
      probability: 0.3,
    },
    {
      name: "Feedback Late",
      type: "feedback_late",
      target_page: "*",
      selector: null,
      params: JSON.stringify({ delay: 1500 }),
      probability: 0.6,
    },

    // C) Search/Navigation Scenarios
    {
      name: "Search Irrelevant",
      type: "search_irrelevant",
      target_page: "/products",
      selector: null,
      params: JSON.stringify({ duration: 5000 }),
      probability: 0.6,
    },
    {
      name: "Facet Reset Once",
      type: "facet_reset_once",
      target_page: "/products",
      selector: null,
      params: JSON.stringify({}),
      probability: 0.3,
    },
    {
      name: "Sort Reset",
      type: "sort_reset",
      target_page: "/products",
      selector: null,
      params: JSON.stringify({}),
      probability: 0.3,
    },

    // D) Data Consistency Scenarios
    {
      name: "Price Change Warning",
      type: "price_change",
      target_page: "/checkout",
      selector: null,
      params: JSON.stringify({ change_percent: 5 }),
      probability: 0.6,
    },

    // E) Cart/Coupon Scenarios
    {
      name: "Coupon Min Spend",
      type: "coupon_min_spend",
      target_page: "/cart",
      selector: null,
      params: JSON.stringify({ min_amount: 500 }),
      probability: 0.6,
    },
    {
      name: "Coupon Expired",
      type: "coupon_expired",
      target_page: "/cart",
      selector: null,
      params: JSON.stringify({}),
      probability: 0.3,
    },

    // F) Payment Scenarios
    {
      name: "3DS Soft Fail",
      type: "3ds_soft_fail",
      target_page: "/checkout",
      selector: null,
      params: JSON.stringify({}),
      probability: 0.6,
    },
    {
      name: "Payment Retry Timeout",
      type: "payment_retry_timeout",
      target_page: "/checkout",
      selector: null,
      params: JSON.stringify({ timeout: 1500 }),
      probability: 0.6,
    },

    // G) Overlay/Attention Scenarios
    {
      name: "Overlay Blocking",
      type: "overlay_blocking",
      target_page: "/",
      selector: null,
      params: JSON.stringify({ duration: 4000 }),
      probability: 0.3,
    },

    // H) Network Scenarios
    {
      name: "Network Jitter",
      type: "network_jitter",
      target_page: "*",
      selector: null,
      params: JSON.stringify({ delay: 500 }),
      probability: 0.6,
    },
  ];

  for (const s of scenarios) {
    await db.run(
      "INSERT INTO scenarios (name, type, target_page, selector, params, probability, enabled) VALUES (?, ?, ?, ?, ?, ?, ?)",
      [s.name, s.type, s.target_page, s.selector, s.params, s.probability, 1]
    );
  }
}

async function seedUsers(db: any) {
  const users = [
    {
      email: 'admin@test.com',
      password: await bcrypt.hash('admin123', 10),
      role: 'admin',
      name: 'Admin User'
    },
    {
      email: 'user@test.com',
      password: await bcrypt.hash('user123', 10),
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