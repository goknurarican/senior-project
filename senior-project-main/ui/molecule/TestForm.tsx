import React from "react";

function TestForm() {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="font-semibold mb-4">Test Form</h3>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          alert("Form submitted!");
        }}
      >
        <input
          type="text"
          placeholder="Coupon Code"
          className="w-full px-3 py-2 border rounded mb-3"
        />
        <button
          type="submit"
          className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700"
        >
          Apply Coupon
        </button>
      </form>
    </div>
  );
}

export default TestForm;
