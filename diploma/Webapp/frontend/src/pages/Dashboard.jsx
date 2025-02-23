import { useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import { useNavigate, Link } from "react-router-dom";

function Dashboard() {
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <div className="container">
      <h1>Welcome, {user?.name}</h1>
      <p>Manage your health with AI-driven predictions.</p>
      <Link to="/prediction">
        <button>Make Prediction</button>
      </Link>
      <button onClick={handleLogout} style={{ marginTop: "10px", backgroundColor: "#dc3545" }}>Logout</button>
    </div>
  );
}

export default Dashboard;
