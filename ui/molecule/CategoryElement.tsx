import Link from "next/link";
import React from "react";
type CategoryElementProps = {
  title: string;
  description: string;
};
function CategoryElement({ title, description }: CategoryElementProps) {
  return (
    <Link
      href="/products?category=electronics"
      className="bg-white p-8 rounded-lg shadow hover:shadow-lg transition-shadow"
    >
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </Link>
  );
}

export default CategoryElement;
