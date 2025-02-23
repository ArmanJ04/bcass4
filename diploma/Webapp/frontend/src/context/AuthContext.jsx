import { createContext, useContext, useState, useEffect } from "react";
import axios from "axios";

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true); // ✅ Added loading state

  // ✅ Move checkAuth OUTSIDE useEffect so it can be used anywhere
  const checkAuth = async () => {
    try {
      const res = await axios.get("http://localhost:5000/api/auth/check-auth", {
        withCredentials: true,
      });
      setUser(res.data.user);
    } catch (error) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkAuth(); // ✅ Runs when the component mounts
  }, []);

  const login = async (email, password) => {
    try {
      const res = await axios.post("http://localhost:5000/api/auth/login", { email, password }, {
        withCredentials: true,
      });
  
      if (res.status === 200 && res.data.user) {
        const userData = res.data.user;
        setUser(userData);
        localStorage.setItem("user", JSON.stringify(userData));
  
        // ✅ Ensure user data refresh
        await checkAuth();
  
        // ✅ Show success alert only when login actually succeeds
        return "success";
      }
    } catch (error) {
      console.error("Login failed:", error.response?.data?.message || error.message);
  
      // ✅ Return error message instead of showing an alert here
      return error.response?.data?.message || "Invalid credentials. Please try again.";
    }
  };  
  const logout = async () => {
    await axios.post("http://localhost:5000/api/auth/logout", {}, {
      withCredentials: true,
    });
    setUser(null);
    localStorage.removeItem("user");
  };

  const updateUser = async (updatedData) => {
    try {
      const res = await axios.put("http://localhost:5000/api/auth/update", updatedData, {
        withCredentials: true,
      });

      setUser(res.data.user);
      localStorage.setItem("user", JSON.stringify(res.data.user));
    } catch (error) {
      console.error("Profile update failed:", error.response?.data?.message || error.message);
    }
  };
  const signup = async (userData) => {
    try {
      const res = await axios.post("http://localhost:5000/api/auth/register", userData, {
        withCredentials: true,
      });
  
      const newUser = res.data.user;
      setUser(newUser); // ✅ Immediately update the state
      localStorage.setItem("user", JSON.stringify(newUser)); // ✅ Store user in localStorage
  
      // ✅ Automatically log in the user after signup
      await login(userData.email, userData.password);
    } catch (error) {
      console.error("Sign-up failed:", error.response?.data?.message || error.message);
    }
  };
  
  
  
  return (
    <AuthContext.Provider value={{ user, signup, login, logout, updateUser, checkAuth, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
