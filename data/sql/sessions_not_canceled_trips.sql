/* # **Title:** Veredelter Session-Datensatz (Elena's Kohorte & Datum-Formatierung)

**Zusammenfassung:** Dieses Skript erstellt einen bereinigten Session-Datensatz für aktive Nutzer.
Es beinhaltet folgende Verbesserungen und Formatierungen:
- Filtert Sessions von Nutzern mit > 7 Sessions nach dem 4. Jan 2023.
- Formatiert alle kritischen Datumsspalten explizit als TIMESTAMP/DATE.
- Berechnet die Session-Dauer in Sekunden.
- Bereinigt negative Werte in der Spalte 'nights'.
- Trennt in der finalen CTE stornierte Reisen und Sessions ohne Trip-ID.

*/

-- 1. CTE: Filtert Sessions nach dem von Elena vorgeschlagenen Zeitraum (nach dem 4. Januar 2023)
WITH sessions_2023 AS (
    SELECT *
    FROM sessions
    -- Konvertiert session_start zu DATE, um die Vergleichssicherheit zu erhöhen
    WHERE CAST(session_start AS DATE) > '2023-01-04'
),

-- 2. CTE: Liefert die IDs aller Nutzer mit mehr als 7 Sessions im Jahr 2023
filtered_users AS (
    SELECT user_id
    FROM sessions_2023
    GROUP BY user_id
    -- COUNT(*) ist hier effizient, da es keine NULL-Werte gibt
    HAVING COUNT(*) > 7
),

-- 3. CTE: Identifiziert alle stornierten Reisen
-- Diese Liste wird später verwendet, um Sessions ohne Storno herauszufiltern
canceled_trips AS (
    SELECT DISTINCT trip_id
    FROM sessions
    WHERE cancellation = TRUE
),

-- 4. CTE: Bildet die Hauptbasis für Sessions (enthält alle Daten für die gefilterten Nutzer)
session_base AS (
    SELECT
        s.session_id,
        s.user_id,
        s.trip_id,
        
        -- DATUMS-FORMATIERUNG
        CAST(s.session_start AS TIMESTAMP) AS session_start,
        CAST(s.session_end AS TIMESTAMP) AS session_end,
        
        -- Feature Engineering: Dauer der Session
        EXTRACT(EPOCH FROM CAST(s.session_end AS TIMESTAMP) - CAST(s.session_start AS TIMESTAMP)) AS session_duration_seconds,
        
        s.page_clicks,
        s.flight_discount,
        s.flight_discount_amount,
        s.hotel_discount,
        s.hotel_discount_amount,
        s.flight_booked,
        s.hotel_booked,
        s.cancellation,
        
        -- DATUMS-FORMATIERUNG (User)
        CAST(u.birthdate AS DATE) AS birthdate,
        CAST(u.sign_up_date AS DATE) AS sign_up_date,

        u.gender,
        u.married,
        u.has_children,
        u.home_country,
        u.home_city,
        u.home_airport,
        u.home_airport_lat,
        u.home_airport_lon,
        
        -- DATUMS-FORMATIERUNG (Flight)
        CAST(f.departure_time AS TIMESTAMP) AS departure_time,
        CAST(f.return_time AS TIMESTAMP) AS return_time,

        f.origin_airport,
        f.destination,
        f.destination_airport,
        f.seats,
        f.return_flight_booked,
        f.checked_bags,
        f.trip_airline,
        f.destination_airport_lat,
        f.destination_airport_lon,
        f.base_fare_usd,
        
        h.hotel_name,
        -- Datenbereinigung: Setzt negative Nächte auf 1
        CASE WHEN h.nights < 0 THEN 1 ELSE h.nights END AS nights, 
        h.rooms,
        
        -- DATUMS-FORMATIERUNG (Hotel)
        CAST(h.check_in_time AS TIMESTAMP) AS check_in_time,
        CAST(h.check_out_time AS TIMESTAMP) AS check_out_time,

        h.hotel_per_room_usd AS hotel_price_per_room_night_usd
    FROM sessions_2023 s
    -- INNER JOIN, da wir nur Sessions von 'filtered_users' wollen
    INNER JOIN filtered_users fu ON s.user_id = fu.user_id 
    LEFT JOIN users u ON s.user_id = u.user_id
    LEFT JOIN flights f ON s.trip_id = f.trip_id
    LEFT JOIN hotels h ON s.trip_id = h.trip_id
),

-- 5. CTE: Filtert alle Sessions ohne Reise-ID und alle stornierten Reisen heraus
not_canceled_trips AS (
    SELECT *
    FROM session_base sb
    -- Filtert 'window shopping' Sessions und Sessions, die keine Buchung darstellen
    WHERE sb.trip_id IS NOT NULL 
    AND NOT EXISTS (
        SELECT 1
        FROM canceled_trips ct
        WHERE ct.trip_id = sb.trip_id
    )
)

-- Finale Abfrage: Alle gültigen, nicht stornierten Sessions
SELECT *
FROM not_canceled_trips;