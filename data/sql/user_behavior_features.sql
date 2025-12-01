/* ==========================================================
# Title: Enriched Session & User Activity Metrics (Elena Cohort)
# Author: [Guy Kaptue]
# Description:
  Builds a complete, enriched user-session dataset based on Elena’s cohort definition.
  The query:
  1. Filters sessions after Jan 4, 2023
  2. Keeps only users with > 7 sessions
  3. Merges user, flight, and hotel data
  4. Cleans and formats date/time columns
  5. Removes canceled and invalid trips
  6. Aggregates metrics on both session and trip levels
  7. Outputs user-level performance metrics
========================================================== */

-- Step 1: Filter sessions after January 4, 2023
WITH sessions_2023 AS (
  SELECT *
  FROM sessions
  WHERE CAST(session_start AS DATE) > '2023-01-04'
),

-- Step 2: Identify active users (>7 sessions)
active_users AS (
  SELECT user_id
  FROM sessions_2023
  GROUP BY user_id
  HAVING COUNT(*) > 7
),

-- Step 3: Build base session table (user, flight, hotel enriched)
session_base AS (
  SELECT
    s.session_id,
    s.user_id,
    s.trip_id,

    -- Timestamp formatting
    CAST(s.session_start AS TIMESTAMP) AS session_start,
    CAST(s.session_end AS TIMESTAMP) AS session_end,

    -- Session duration in seconds
    EXTRACT(EPOCH FROM (CAST(s.session_end AS TIMESTAMP) - CAST(s.session_start AS TIMESTAMP))) AS session_duration_seconds,

    -- Core session data
    s.page_clicks,
    s.flight_discount,
    s.flight_discount_amount,
    s.hotel_discount,
    s.hotel_discount_amount,
    s.flight_booked,
    s.hotel_booked,
    s.cancellation,

    -- User data
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

    -- Flight data
    f.origin_airport,
    f.destination,
    f.destination_airport,
    f.seats,
    f.return_flight_booked,
    CAST(f.departure_time AS TIMESTAMP) AS departure_time,
    CAST(f.return_time AS TIMESTAMP) AS return_time,
    f.checked_bags,
    f.trip_airline,
    f.destination_airport_lat,
    f.destination_airport_lon,
    f.base_fare_usd,

    -- Hotel data
    h.hotel_name,
    CASE WHEN h.nights < 0 THEN 1 ELSE h.nights END AS nights,
    h.rooms,
    CAST(h.check_in_time AS TIMESTAMP) AS check_in_time,
    CAST(h.check_out_time AS TIMESTAMP) AS check_out_time,
    h.hotel_per_room_usd AS hotel_price_per_room_night_usd

  FROM sessions_2023 s
  INNER JOIN active_users au ON s.user_id = au.user_id
  LEFT JOIN users u ON s.user_id = u.user_id
  LEFT JOIN flights f ON s.trip_id = f.trip_id
  LEFT JOIN hotels h ON s.trip_id = h.trip_id
),

-- Step 4: Identify canceled trips
canceled_trips AS (
  SELECT DISTINCT trip_id
  FROM session_base
  WHERE cancellation = TRUE
),

-- Step 5: Keep valid, non-canceled trips only
valid_trips AS (
  SELECT *
  FROM session_base
  WHERE trip_id IS NOT NULL
    AND trip_id NOT IN (SELECT trip_id FROM canceled_trips)
),

-- Step 6: Session-level metrics per user
user_session_metrics AS (
  SELECT
    user_id,
    COUNT(DISTINCT session_id) AS total_sessions,
    SUM(page_clicks) AS total_page_clicks,
    ROUND(AVG(page_clicks), 2) AS avg_clicks_per_session,
    ROUND(AVG(session_duration_seconds), 2) AS avg_session_duration_seconds
  FROM session_base
  GROUP BY user_id
),

-- Step 7: Trip-level metrics per user
user_trip_metrics AS (
  SELECT
    user_id,
    COUNT(DISTINCT trip_id) AS total_trips,
    SUM(
      CASE
        WHEN flight_booked = TRUE AND return_flight_booked = TRUE THEN 2
        WHEN flight_booked = TRUE THEN 1
        ELSE 0
      END
    ) AS total_flights,
    ROUND(AVG(DATE_PART('day', departure_time - session_end)), 2) AS avg_days_between_booking_and_departure
  FROM valid_trips
  GROUP BY user_id
)

-- Final Output: Combined user-level summary
SELECT
  s.user_id,
  s.total_sessions,
  s.total_page_clicks,
  s.avg_clicks_per_session,
  s.avg_session_duration_seconds,
  t.total_trips,
  t.total_flights,
  t.avg_days_between_booking_and_departure
FROM user_session_metrics s
LEFT JOIN user_trip_metrics t ON s.user_id = t.user_id
ORDER BY s.total_sessions DESC;

