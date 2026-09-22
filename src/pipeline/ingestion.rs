use crate::domain::entities::{ApplicationRecord, CreditHistoryRecord, CustomerId, MonthStatus};
use crate::domain::traits::AppResult;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Ingestor de datasets crudos con buffers eficientes y validación tipada.
pub struct CsvIngestor;

impl CsvIngestor {
    /// Ingesta application_record.csv y deduplica por ID (manteniendo el último registro coherente).
    pub fn load_applications<P: AsRef<Path>>(path: P) -> AppResult<HashMap<CustomerId, ApplicationRecord>> {
        let file = File::open(path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(BufReader::with_capacity(1024 * 1024, file));

        let mut map = HashMap::with_capacity(450_000);

        for result in rdr.records() {
            let record = result?;
            if record.len() < 18 {
                continue;
            }

            let id: CustomerId = match record[0].parse() {
                Ok(val) => val,
                Err(_) => continue,
            };

            let gender = record[1].to_string();
            let own_car = &record[2] == "Y";
            let own_realty = &record[3] == "Y";
            let children_count: i32 = record[4].parse().unwrap_or(0);
            let total_income: f64 = record[5].parse().unwrap_or(0.0);
            let income_type = record[6].to_string();
            let education_type = record[7].to_string();
            let family_status = record[8].to_string();
            let housing_type = record[9].to_string();
            let days_birth: i32 = record[10].parse().unwrap_or(0);
            let days_employed: i32 = record[11].parse().unwrap_or(0);
            let has_mobile = &record[12] == "1";
            let has_work_phone = &record[13] == "1";
            let has_phone = &record[14] == "1";
            let has_email = &record[15] == "1";
            let occupation_type = if record[16].trim().is_empty() {
                None
            } else {
                Some(record[16].to_string())
            };
            let family_members_count: f64 = record[17].parse().unwrap_or(1.0);

            let app = ApplicationRecord {
                id,
                gender,
                own_car,
                own_realty,
                children_count,
                total_income,
                income_type,
                education_type,
                family_status,
                housing_type,
                days_birth,
                days_employed,
                has_mobile,
                has_work_phone,
                has_phone,
                has_email,
                occupation_type,
                family_members_count,
            };

            map.insert(id, app);
        }

        Ok(map)
    }

    /// Ingesta credit_record.csv y agrupa los registros mensuales por CustomerId.
    pub fn load_credit_records<P: AsRef<Path>>(
        path: P,
    ) -> AppResult<HashMap<CustomerId, Vec<CreditHistoryRecord>>> {
        let file = File::open(path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(BufReader::with_capacity(1024 * 1024, file));

        let mut map: HashMap<CustomerId, Vec<CreditHistoryRecord>> = HashMap::with_capacity(50_000);

        for result in rdr.records() {
            let record = result?;
            if record.len() < 3 {
                continue;
            }

            let id: CustomerId = match record[0].parse() {
                Ok(val) => val,
                Err(_) => continue,
            };

            let months_balance: i32 = match record[1].parse() {
                Ok(val) => val,
                Err(_) => continue,
            };

            let status = MonthStatus::parse(&record[2]);

            let item = CreditHistoryRecord {
                id,
                months_balance,
                status,
            };

            map.entry(id).or_default().push(item);
        }

        Ok(map)
    }
}
