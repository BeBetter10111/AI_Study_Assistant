import { useState } from "react";
import Icon from "../../components/Icons";
import { getErrorMessage, updateProfile } from "../../services/api";
import { displayName, initials, roleLabel } from "../../utils/user";

const joinedText = (iso) => {
  const date = iso ? new Date(iso) : null;
  if (!date || Number.isNaN(date.getTime())) return null;
  return date.toLocaleDateString("vi-VN", { month: "long", year: "numeric" });
};

export default function ProfilePage({ profile, onUpdated }) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({ username: "", full_name: "", phone: "" });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState(false);

  if (!profile) {
    return (
      <div className="page">
        <p className="page-sub">Đang tải hồ sơ…</p>
      </div>
    );
  }

  const name = displayName(profile);
  const joined = joinedText(profile.created_at);

  const startEdit = () => {
    setForm({
      username: profile.username || "",
      full_name: profile.full_name || "",
      phone: profile.phone || "",
    });
    setError(null);
    setSaved(false);
    setEditing(true);
  };

  const cancelEdit = () => {
    setEditing(false);
    setError(null);
  };

  const setField = (field) => (e) => setForm((prev) => ({ ...prev, [field]: e.target.value }));

  const handleSave = async (e) => {
    e.preventDefault();
    if (saving) return;
    if (form.username.trim().length < 3) {
      setError("Tên đăng nhập cần ít nhất 3 ký tự.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const res = await updateProfile({
        username: form.username.trim(),
        full_name: form.full_name.trim(),
        phone: form.phone.trim(),
      });
      onUpdated(res.data);
      setEditing(false);
      setSaved(true);
    } catch (err) {
      setError(getErrorMessage(err, "Không thể lưu thay đổi. Vui lòng thử lại."));
    } finally {
      setSaving(false);
    }
  };

  // Chế độ xem: hiển thị đúng các ô nhập ở dạng chỉ đọc, để bố cục không nhảy khi bấm Chỉnh sửa.
  const value = (field) => (editing ? form[field] : profile[field] || "");

  return (
    <div className="page">
      <header className="page-head">
        <h1 className="page-title">Hồ sơ cá nhân</h1>
        <p className="page-sub">Quản lý thông tin tài khoản của bạn.</p>
      </header>

      <section className="card card-pad">
        <div className="profile-head">
          <span className="avatar avatar--lg">{initials(name)}</span>
          <div className="profile-id">
            <div className="profile-name">
              <span>{name}</span>
              {profile.role && <span className="badge">{roleLabel(profile.role)}</span>}
            </div>
            <div className="profile-mail">
              <Icon name="mail" size={15} />
              <span>{profile.email}</span>
            </div>
          </div>
          {!editing && (
            <button type="button" className="btn btn--primary" onClick={startEdit}>
              Chỉnh sửa
            </button>
          )}
        </div>
      </section>

      <form className="card card-pad" onSubmit={handleSave} noValidate>
        <h2 className="card-title">Thông tin cá nhân</h2>
        <p className="card-sub">Cập nhật tên và số điện thoại của bạn.</p>

        {saved && !editing && (
          <p className="note note--ok" role="status" style={{ marginTop: 16 }}>
            Đã lưu thay đổi.
          </p>
        )}

        <div className="form-grid" style={{ marginTop: 20 }}>
          <div className="field">
            <label className="label" htmlFor="pf-username">
              Tên đăng nhập
            </label>
            <input
              id="pf-username"
              className="input"
              value={value("username")}
              onChange={setField("username")}
              readOnly={!editing}
              autoComplete="username"
              maxLength={150}
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="pf-fullname">
              Họ và tên
            </label>
            <input
              id="pf-fullname"
              className="input"
              value={value("full_name")}
              onChange={setField("full_name")}
              readOnly={!editing}
              placeholder={editing ? "Nguyễn Văn A" : "Chưa cập nhật"}
              autoComplete="name"
              maxLength={255}
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="pf-email">
              Email
            </label>
            <input id="pf-email" className="input" value={profile.email} readOnly />
            {editing && (
              <span className="field-hint">Email dùng để khôi phục mật khẩu nên không thể sửa ở đây.</span>
            )}
          </div>
          <div className="field">
            <label className="label" htmlFor="pf-phone">
              Số điện thoại
            </label>
            <input
              id="pf-phone"
              className="input"
              type="tel"
              value={value("phone")}
              onChange={setField("phone")}
              readOnly={!editing}
              placeholder={editing ? "09xx xxx xxx" : "Chưa cập nhật"}
              autoComplete="tel"
              maxLength={30}
            />
          </div>
        </div>

        {error && (
          <p className="note note--err" role="alert" style={{ marginTop: 16 }}>
            {error}
          </p>
        )}

        {editing && (
          <div className="actions" style={{ marginTop: 20 }}>
            <button type="submit" className="btn btn--primary" disabled={saving}>
              {saving ? "Đang lưu…" : "Lưu thay đổi"}
            </button>
            <button type="button" className="btn" onClick={cancelEdit} disabled={saving}>
              Hủy
            </button>
          </div>
        )}

        {joined && !editing && (
          <p className="field-hint" style={{ marginTop: 20 }}>
            Tham gia từ {joined}
          </p>
        )}
      </form>
    </div>
  );
}
