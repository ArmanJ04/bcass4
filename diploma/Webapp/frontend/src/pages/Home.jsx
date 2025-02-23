import { Link } from "react-router-dom";

function Home() {
  return (
    <div className="container">
      <h1>Welcome to CVD Prediction</h1>
      <p>Your health matters. Predict cardiovascular disease risk easily.</p>
      <Link to="/login">
        <button>Get Started</button>
      </Link>
    </div>
  );
}

export default Home;
