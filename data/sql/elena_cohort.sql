/* 
# **Title:** Enriched Session Dataset Based on Elena’s Cohort Definition (Date Formatting Added)

**Summary:** 
Diese SQL-Logik erstellt ein verfeinertes Session-Dataset in vier Schritten, basierend auf Elena’s Kohortendefinition:
1. **sessions_2023** – Filtert Sessions nach dem 4. Januar 2023  
2. **filtered_users** – Wählt aktive Nutzer mit mehr als 7 Sessions  
3. **session_base** – Joint Sessions mit User-, Flug- und Hotel-Daten  
4. **Final Output** – Gibt alle angereicherten Sessions mit korrektem Datumsformat zurück  
*/
-- This CTE pre-limits our sessions to the suggested timeframe (After Jan 4 2023)
WITH sessions_2023 AS (
  SELECT
    user_id,
    session_id,
    trip_id,
    session_start,
    session_end,
    page_clicks,
    flight_discount,
    flight_discount_amount,
    hotel_discount,
    hotel_discount_amount,
    flight_booked,
    hotel_booked,
    cancellation
  FROM sessions
  WHERE session_start > '2023-01-04'
),

-- This CTE returns the ids of all users with more than 7 sessions in 2023
filtered_users AS (
  SELECT user_id
  FROM sessions_2023
  GROUP BY user_id
  HAVING COUNT(session_id) > 7
),

-- This CTE joins the sessions with user, flight, and hotel data
session_base AS (
  SELECT
    s.session_id,
    s.user_id,
    s.trip_id,
    s.session_start,
    s.session_end,
    s.page_clicks,
    s.flight_discount,
    s.flight_discount_amount,
    s.hotel_discount,
    s.hotel_discount_amount,
    s.flight_booked,
    s.hotel_booked,
    s.cancellation,
    u.birthdate,
    u.gender,
    u.married,
    u.has_children,
    u.home_country,
    u.home_city,
    u.home_airport,
    u.home_airport_lat,
    u.home_airport_lon,
    u.sign_up_date,
    f.origin_airport,
    f.destination,
    f.destination_airport,
    f.seats,
    f.return_flight_booked,
    f.departure_time,
    f.return_time,
    f.checked_bags,
    f.trip_airline,
    f.destination_airport_lat,
    f.destination_airport_lon,
    f.base_fare_usd,
    h.hotel_name,
    h.nights,
    h.rooms,
    h.check_in_time,
    h.check_out_time,
    h.hotel_per_room_usd AS hotel_price_per_room_night_usd
  FROM sessions_2023 s
  LEFT JOIN users u
    ON s.user_id = u.user_id
  LEFT JOIN flights f
    ON s.trip_id = f.trip_id
  LEFT JOIN hotels h
    ON s.trip_id = h.trip_id
  WHERE
    s.user_id IN (
      SELECT user_id
      FROM filtered_users
    )
)

-- Final result set
SELECT *
FROM session_base;