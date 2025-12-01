SELECT
  DISTINCT trip_id
FROM sessions 
WHERE cancellation = TRUE;