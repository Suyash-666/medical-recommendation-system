# React Frontend

This frontend is now the primary UI for the Flask app.

## Run in development

1. `cd frontend`
2. `npm install`
3. `npm run dev`

## Build for Flask

1. `cd frontend`
2. `npm run build`

Flask serves `frontend/dist/index.html` and `frontend/dist/assets/*`.

## Backend API used by React

- `POST /api/login`
- `POST /api/signup`
- `POST /api/logout`
- `GET /api/session`
- `GET /api/dashboard`
- `POST /api/predict`
- `GET /api/history`
- `GET/POST /api/health-profile`
- `GET /api/health-tips`
- `GET/POST /api/emergency-sos`
- `POST /api/specialist-finder`
- `POST /api/lab-upload`
- `GET /api/lab-analysis/:reportId`
- `GET/POST /api/reminders`
- `POST /api/delete-reminder/:id`
- `GET /api/notifications`
- `POST /api/mark-notification-read/:id`
- `POST /api/delete-notification/:id`
- `POST /api/mark-all-read`
