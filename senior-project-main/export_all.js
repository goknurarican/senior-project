const sqlite3 = require("sqlite3").verbose();
const fs = require("fs");
const fastcsv = require("fast-csv");

const DB_PATH = "./experiment.db";   
const db = new sqlite3.Database(DB_PATH);

// 1) Tüm tablo isimlerini çek
db.all(
    `SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';`,
    [],
    (err, tables) => {
        if (err) {
            console.error("Tablolar alınamadı:", err);
            return;
        }

        console.log("Bulunan tablolar:", tables.map(t => t.name));

        // 2) Her tablo için CSV üret
        tables.forEach(table => {
            const tableName = table.name;
            const fileName = `${tableName}.csv`;

            db.all(`SELECT * FROM ${tableName}`, [], (err, rows) => {
                if (err) {
                    console.error(`${tableName} okunurken hata:`, err);
                    return;
                }

                const ws = fs.createWriteStream(fileName);

                fastcsv
                    .write(rows, { headers: true })
                    .on("finish", () => {
                        console.log(`✓ ${fileName} oluşturuldu (${rows.length} satır)`);
                    })
                    .pipe(ws);
            });
        });
    }
);
