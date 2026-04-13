// pages/signup.tsx
import { useState } from "react";
import { NextRouter, useRouter } from "next/router";
import Link from "next/link";
import { User } from "../types/types";

export default function SignUp() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    age: "",
    gender: "",
    handedness: "right",
    vision_correction: "none",
  });
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<boolean>(false);
  const router: NextRouter = useRouter();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess(false);

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters long");
      return;
    }

    try {
      const res = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name:             formData.name,
          email:            formData.email,
          password:         formData.password,
          age:              formData.age ? parseInt(formData.age) : null,
          gender:           formData.gender || null,
          handedness:       formData.handedness,
          vision_correction: formData.vision_correction,
        }),
      });

      const data: User = await res.json();
      if (res.ok) {
        await fetch("/api/auth/me");
        setSuccess(true);
        setTimeout(() => {
          router.push("/").then(() => window.location.reload());
        }, 2000);
      } else {
        setError((data as any).error || "Sign up failed");
      }
    } catch (err) {
      setError("An error occurred. Please try again.");
    }
  };

  const inputCls = "mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-sm";
  const labelCls = "block text-sm font-medium text-gray-700";

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900">Create Account</h2>
            <p className="mt-2 text-sm text-gray-600">Sign up to get started</p>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded text-sm">
              {error}
            </div>
          )}
          {success && (
            <div className="mb-4 p-3 bg-green-100 border border-green-400 text-green-700 rounded text-sm">
              Account created successfully! Redirecting...
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Name */}
            <div>
              <label className={labelCls}>Full Name</label>
              <input type="text" name="name" required value={formData.name}
                onChange={handleChange} className={inputCls} placeholder="John Doe" />
            </div>

            {/* Email */}
            <div>
              <label className={labelCls}>Email</label>
              <input type="email" name="email" required value={formData.email}
                onChange={handleChange} className={inputCls} placeholder="john@example.com" />
            </div>

            {/* Password */}
            <div>
              <label className={labelCls}>Password</label>
              <input type="password" name="password" required value={formData.password}
                onChange={handleChange} className={inputCls} placeholder="At least 6 characters" />
            </div>

            {/* Confirm Password */}
            <div>
              <label className={labelCls}>Confirm Password</label>
              <input type="password" name="confirmPassword" required value={formData.confirmPassword}
                onChange={handleChange} className={inputCls} placeholder="Re-enter your password" />
            </div>

            <hr className="border-gray-200" />
            <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">
              Participant Information (for research purposes)
            </p>

            {/* Age */}
            <div>
              <label className={labelCls}>Age</label>
              <input type="number" name="age" min="18" max="99" value={formData.age}
                onChange={handleChange} className={inputCls} placeholder="e.g. 24" />
            </div>

            {/* Gender */}
            <div>
              <label className={labelCls}>Gender</label>
              <select name="gender" value={formData.gender} onChange={handleChange} className={inputCls}>
                <option value="">Prefer not to say</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="non_binary">Non-binary</option>
                <option value="other">Other</option>
              </select>
            </div>

            {/* Handedness */}
            <div>
              <label className={labelCls}>Dominant Hand</label>
              <select name="handedness" value={formData.handedness} onChange={handleChange} className={inputCls}>
                <option value="right">Right</option>
                <option value="left">Left</option>
                <option value="ambidextrous">Ambidextrous</option>
              </select>
            </div>

            {/* Vision correction */}
            <div>
              <label className={labelCls}>Vision Correction</label>
              <select name="vision_correction" value={formData.vision_correction} onChange={handleChange} className={inputCls}>
                <option value="none">None</option>
                <option value="glasses">Glasses</option>
                <option value="contact_lenses">Contact Lenses</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={success}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {success ? "Account Created!" : "Sign up"}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600">
              Already have an account?{" "}
              <Link href="/login" className="font-medium text-blue-600 hover:text-blue-500">
                Sign in here
              </Link>
            </p>
          </div>
          <div className="mt-4 text-center">
            <Link href="/" className="text-sm text-blue-600 hover:text-blue-500">
              Continue as guest →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
