# employee-info-api

This is a tiny MuleSoft API I put together to look up employee details by ID. It exposes one endpoint, `GET /employees/{id}`, which hits a MySQL table under the hood and returns a clean JSON response with the employee’s name, department, and salary. It uses a basic HTTP Listener, a Database Connector configured for MySQL, and a small DataWeave transform to shape the result.

It’s intentionally simple and beginner‑friendly: easy to run locally in Anypoint Studio, easy to read, and easy to extend later (add more fields, more endpoints, validation, etc.). If you’ve been meaning to get comfortable with Mule flows, DB queries, and returning JSON responses, this is a nice, no‑surprises starting point.