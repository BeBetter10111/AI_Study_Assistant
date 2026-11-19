import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
});

export const setAuthToken = (token) => {
  if (token) {
    api.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common.Authorization;
  }
};

let unauthorizedHandler = null;
export const setUnauthorizedHandler = (handler) => {
  unauthorizedHandler = handler;
};

const notifyUnauthorized = () => {
  if (api.defaults.headers.common.Authorization && unauthorizedHandler) unauthorizedHandler();
};

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const url = error?.config?.url || "";
    if (error?.response?.status === 401 && !url.startsWith("/auth/")) notifyUnauthorized();
    return Promise.reject(error);
  }
);

// FastAPI returns `detail` as a string (HTTPException) or an ARRAY (422 validation error). Rendering an array
// of objects directly in JSX crashes React -> blank page, so always convert it to a string.
const formatDetail = (detail) => {
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => d?.msg || JSON.stringify(d)).join("; ");
  }
  return null;
};

export const getErrorMessage = (err, fallback) => {
  if (err && !err.response && err.request) {
    return (
      `Cannot reach the server at ${api.defaults.baseURL || "(VITE_API_BASE_URL is not set)"}. ` +
      "Please check that the backend is running."
    );
  }
  return formatDetail(err?.response?.data?.detail) || fallback;
};

export const registerUser = (data) =>
  api.post("/auth/register", data);

export const loginUser = (identifier, password) =>
  api.post("/auth/login", { identifier, password });

export const forgotPassword = (email) =>
  api.post("/auth/forgot-password", { email });

export const verifyResetCode = (email, code) =>
  api.post("/auth/verify-reset-code", { email, code });

export const resetPassword = (resetToken, newPassword, confirmPassword) =>
  api.post("/auth/reset-password", {
    reset_token: resetToken,
    new_password: newPassword,
    confirm_password: confirmPassword,
  });

export const getMe = () => api.get("/auth/me");

export const updateProfile = (data) => api.patch("/auth/me", data);

export const changePassword = (currentPassword, newPassword, confirmPassword) =>
  api.post("/auth/change-password", {
    current_password: currentPassword,
    new_password: newPassword,
    confirm_password: confirmPassword,
  });

// Documents and conversation history are stored on the server, so lists can be restored after a page reload.
export const listDocuments = () => api.get("/documents");

export const deleteDocument = (id) => api.delete(`/documents/${id}`);

export const listConversations = () => api.get("/conversations");

export const createConversation = (documentId = null) =>
  api.post("/conversations", { document_id: documentId });

export const getConversation = (id) => api.get(`/conversations/${id}`);

export const renameConversation = (id, title) => api.patch(`/conversations/${id}`, { title });

export const deleteConversation = (id) => api.delete(`/conversations/${id}`);

export const uploadDocument = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/documents/upload", form);
};

export const getDocumentMindmap = (documentId) =>
  api.get(`/documents/${documentId}/mindmap`);

export const getDocumentStatus = (documentId) =>
  api.get(`/documents/${documentId}/status`);

export const askQuestion = (question, documentId = null, topK = 4) =>
  api.post("/chat/query", {
    question,
    document_id: documentId,
    top_k: topK,
  });

export const generateQuiz = (topic, numQuestions = 10, documentId = null) =>
  api.post("/quiz/generate", {
    topic,
    num_questions: numQuestions,
    document_id: documentId,
  });

export const submitQuizResult = (quizId, result) =>
  api.post(`/quiz/${quizId}/submit`, result);

// Streams the RAG pipeline as Server-Sent Events so the UI can show each real
// step ("Searching your document", ...). axios cannot stream in the browser,
// so this uses fetch. Calls onEvent({type: "step"|"result"|"error", ...}).
const parseSseFrame = (frame) => {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n");
  if (!data) return null;
  try {
    return JSON.parse(data);
  } catch {
    console.warn("Ignoring malformed SSE frame:", data.slice(0, 120));
    return null;
  }
};

// options: { topK, history: [{role, content}], signal: AbortSignal }
// Pass `signal` to cancel mid-request ("Stop" button): the server detects the closed connection and stops generating the answer.
export const askQuestionStream = async (
  question,
  documentId,
  onEvent,
  { topK = 4, history = [], signal, conversationId = null } = {}
) => {
  const response = await fetch(`${api.defaults.baseURL || ""}/chat/query/stream`, {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      ...(api.defaults.headers.common.Authorization
        ? { Authorization: api.defaults.headers.common.Authorization }
        : {}),
    },
    body: JSON.stringify({
      question,
      document_id: documentId,
      top_k: topK,
      history,
      // When conversation_id is present, the server saves the message and loads history from the DB (ignoring `history` above).
      ...(conversationId ? { conversation_id: conversationId } : {}),
    }),
  });

  if (response.status === 401) notifyUnauthorized();
  if (!response.ok || !response.body) {
    let detail = "An error occurred while connecting to the AI assistant.";
    try {
      detail = formatDetail((await response.json()).detail) || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const flush = (frames) => {
    for (const frame of frames) {
      const event = parseSseFrame(frame);
      if (event) onEvent(event);
    }
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
    const frames = buffer.split("\n\n");
    buffer = frames.pop();
    flush(frames);
  }
  buffer += decoder.decode();
  if (buffer.trim()) flush([buffer]); // The final frame has no trailing blank line.
};

// Exam-prep mode
export const generateExam = (data) => api.post("/exams/generate", data, { timeout: 300000 });
export const listExams = () => api.get("/exams");
export const getExam = (id) => api.get(`/exams/${id}`);
export const submitExam = (id, answers, durationS) =>
  api.post(`/exams/${id}/submit`, { answers, duration_s: durationS });
export const deleteExam = (id) => api.delete(`/exams/${id}`);
export const getWeakTopics = () => api.get("/exams/report/weak-topics");

// Lecturer mode
export const generateBankItems = (data) => api.post("/lecturer/bank/generate", data, { timeout: 300000 });
export const listBankItems = (params) => api.get("/lecturer/bank", { params });
export const patchBankItem = (id, data) => api.patch(`/lecturer/bank/${id}`, data);
export const deleteBankItem = (id) => api.delete(`/lecturer/bank/${id}`);
export const gradeBankAnswer = (id, answer) => api.post(`/lecturer/bank/${id}/grade`, { answer }, { timeout: 180000 });
export const composePaper = (data) => api.post("/lecturer/papers", data);
export const listPapers = () => api.get("/lecturer/papers");
export const deletePaper = (id) => api.delete(`/lecturer/papers/${id}`);
export const downloadPaper = (id, format, includeAnswers) =>
  api.get(`/lecturer/papers/${id}/export`, {
    params: { format, include_answers: includeAnswers },
    responseType: "blob",
  });

// Speaking practice (Whisper)
export const getSpeechStatus = () => api.get("/speech/status");
export const practiceSpeech = (blob, filename, language, referenceText) => {
  const form = new FormData();
  form.append("file", blob, filename);
  form.append("language", language);
  if (referenceText) form.append("reference_text", referenceText);
  return api.post("/speech/practice", form, { timeout: 180000 });
};
