import React, { useEffect, useState } from 'react';
import { Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom';
import { toast } from 'react-toastify';
import { apiGet, apiPost } from './api';
import RequireAuth from './components/RequireAuth';

import {
  LandingPage,
  LoginPage,
  SignupPage,
  DashboardPage,
  PredictPage,
  ResultPage,
  HistoryPage,
  HealthProfilePage,
  HealthTipsPage,
  PharmacyLocatorPage,
  SpecialistFinderPage,
  LabUploadPage,
  LabAnalysisPage,
  RemindersPage,
  NotificationsPage,
  EmergencySosPage,
} from '../../react_templates/index';

function LoginRoute() {
  const navigate = useNavigate();
  return (
    <LoginPage
      onSubmit={async (form) => {
        try {
          await apiPost('/api/login', form);
          toast.success('Login successful');
          navigate('/dashboard');
        } catch (e) {
          toast.error(e.message);
        }
      }}
    />
  );
}

function SignupRoute() {
  const navigate = useNavigate();
  return (
    <SignupPage
      onSubmit={async (form) => {
        try {
          await apiPost('/api/signup', form);
          toast.success('Account created. Please login.');
          navigate('/login');
        } catch (e) {
          toast.error(e.message);
        }
      }}
    />
  );
}

function DashboardRoute() {
  const [username, setUsername] = useState('User');
  const [records, setRecords] = useState([]);
  useEffect(() => {
    apiGet('/api/dashboard')
      .then((d) => {
        setUsername(d.username || 'User');
        setRecords(d.records || []);
      })
      .catch((e) => toast.error(e.message));
  }, []);
  return <DashboardPage username={username} records={records} />;
}

function PredictRoute() {
  const navigate = useNavigate();
  return (
    <PredictPage
      onSubmit={async (form) => {
        try {
          const result = await apiPost('/api/predict', form);
          sessionStorage.setItem('lastPrediction', JSON.stringify(result));
          toast.success('Prediction complete');
          navigate('/result');
        } catch (e) {
          toast.error(e.message);
        }
      }}
    />
  );
}

function ResultRoute() {
  const [result, setResult] = useState(null);
  useEffect(() => {
    const raw = sessionStorage.getItem('lastPrediction');
    if (raw) setResult(JSON.parse(raw));
  }, []);
  if (!result) return <Navigate to="/predict" replace />;
  return <ResultPage {...result} />;
}

function HistoryRoute() {
  const [history, setHistory] = useState([]);
  useEffect(() => {
    apiGet('/api/history').then((d) => setHistory(d.history || [])).catch((e) => toast.error(e.message));
  }, []);
  return <HistoryPage history={history} />;
}

function HealthProfileRoute() {
  const [user, setUser] = useState({});
  useEffect(() => {
    apiGet('/api/health-profile').then((d) => setUser(d.user || {})).catch((e) => toast.error(e.message));
  }, []);
  return (
    <HealthProfilePage
      user={user}
      onSubmit={async (form) => {
        try {
          await apiPost('/api/health-profile', form);
          toast.success('Profile updated');
        } catch (e) {
          toast.error(e.message);
        }
      }}
    />
  );
}

function HealthTipsRoute() {
  const [tips, setTips] = useState([]);
  useEffect(() => {
    apiGet('/api/health-tips').then((d) => setTips(d.tips || [])).catch((e) => toast.error(e.message));
  }, []);
  return <HealthTipsPage tips={tips} />;
}

function EmergencySosRoute() {
  const [user, setUser] = useState({});
  useEffect(() => {
    apiGet('/api/emergency-sos').then((d) => setUser(d.user || {})).catch((e) => toast.error(e.message));
  }, []);
  return (
    <EmergencySosPage
      user={user}
      onSubmit={async (form) => {
        try {
          await apiPost('/api/emergency-sos', form);
          toast.success('Emergency info updated');
        } catch (e) {
          toast.error(e.message);
        }
      }}
    />
  );
}

function SpecialistFinderRoute() {
  const [specialists, setSpecialists] = useState([]);
  const [location, setLocation] = useState('');
  return (
    <SpecialistFinderPage
      specialists={specialists}
      location={location}
      onSubmit={async (form) => {
        try {
          const res = await apiPost('/api/specialist-finder', form);
          setSpecialists(res.specialists || []);
          setLocation(res.location || '');
          toast.success('Specialists loaded');
        } catch (e) {
          toast.error(e.message);
        }
      }}
    />
  );
}

function LabUploadRoute() {
  const navigate = useNavigate();
  return (
    <LabUploadPage
      onSubmit={async (form) => {
        try {
          const res = await apiPost('/api/lab-upload', form);
          toast.success('Lab report uploaded');
          navigate(`/lab-analysis/${res.report_id}`);
        } catch (e) {
          toast.error(e.message);
        }
      }}
    />
  );
}

function LabAnalysisRoute() {
  const { reportId } = useParams();
  const [data, setData] = useState(null);
  useEffect(() => {
    apiGet(`/api/lab-analysis/${reportId}`)
      .then((d) => setData(d))
      .catch((e) => toast.error(e.message));
  }, [reportId]);
  if (!data) return <div style={{ padding: 20 }}>Loading analysis...</div>;
  return <LabAnalysisPage {...data} />;
}

function RemindersRoute() {
  const [reminders, setReminders] = useState([]);
  const refresh = () => apiGet('/api/reminders').then((d) => setReminders(d.reminders || [])).catch((e) => toast.error(e.message));
  useEffect(() => { refresh(); }, []);
  return (
    <RemindersPage
      reminders={reminders}
      onAddReminder={async (form) => {
        try {
          await apiPost('/api/reminders', form);
          toast.success('Reminder added');
          refresh();
        } catch (e) {
          toast.error(e.message);
        }
      }}
      onDeleteReminder={async (id) => {
        try {
          await apiPost(`/api/delete-reminder/${id}`);
          toast.success('Reminder deleted');
          refresh();
        } catch (e) {
          toast.error(e.message);
        }
      }}
    />
  );
}

function NotificationsRoute() {
  const [data, setData] = useState({ notifications: [], current_page: 1, total_pages: 1 });
  useEffect(() => {
    apiGet('/api/notifications').then((d) => setData(d)).catch((e) => toast.error(e.message));
  }, []);
  return <NotificationsPage notifications={data.notifications} current_page={data.current_page} total_pages={data.total_pages} />;
}

function LogoutRoute() {
  const navigate = useNavigate();
  useEffect(() => {
    apiPost('/api/logout').finally(() => navigate('/login'));
  }, [navigate]);
  return <div style={{ padding: 20 }}>Logging out...</div>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/login" element={<LoginRoute />} />
      <Route path="/signup" element={<SignupRoute />} />
      <Route path="/logout" element={<LogoutRoute />} />

      <Route path="/dashboard" element={<RequireAuth><DashboardRoute /></RequireAuth>} />
      <Route path="/predict" element={<RequireAuth><PredictRoute /></RequireAuth>} />
      <Route path="/result" element={<RequireAuth><ResultRoute /></RequireAuth>} />
      <Route path="/history" element={<RequireAuth><HistoryRoute /></RequireAuth>} />
      <Route path="/health-profile" element={<RequireAuth><HealthProfileRoute /></RequireAuth>} />
      <Route path="/health-tips" element={<RequireAuth><HealthTipsRoute /></RequireAuth>} />
      <Route path="/pharmacy-locator" element={<RequireAuth><PharmacyLocatorPage /></RequireAuth>} />
      <Route path="/specialist-finder" element={<RequireAuth><SpecialistFinderRoute /></RequireAuth>} />
      <Route path="/lab-upload" element={<RequireAuth><LabUploadRoute /></RequireAuth>} />
      <Route path="/lab-analysis/:reportId" element={<RequireAuth><LabAnalysisRoute /></RequireAuth>} />
      <Route path="/reminders" element={<RequireAuth><RemindersRoute /></RequireAuth>} />
      <Route path="/notifications" element={<RequireAuth><NotificationsRoute /></RequireAuth>} />
      <Route path="/emergency-sos" element={<RequireAuth><EmergencySosRoute /></RequireAuth>} />

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
