// scripts/reset-db.js
const fs = require("fs");
const path = require("path");

const dbPath = path.join(process.cwd(), "experiment.db");

if (fs.existsSync(dbPath)) {
  fs.unlinkSync(dbPath);
  console.log("✅ Database deleted successfully");
} else {
  console.log("ℹ️ No database file found");
}

console.log("📝 Database will be recreated on next app start");
console.log("\nTest accounts:");
console.log("Admin: admin@test.com / admin123");
console.log("User: user@test.com / user123");