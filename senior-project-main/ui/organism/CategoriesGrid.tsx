import React from "react";
import CategoryElement from "../molecule/CategoryElement";

function CategoriesGrid() {
  return (
    <div className="grid grid-cols-2 gap-6">
      <CategoryElement
        title={"Electronics"}
        description={"Latest tech gadgets and accessories"}
      />
      <CategoryElement
        title={"Home & Office"}
        description={"Furniture and home essentials"}
      />
    </div>
  );
}

export default CategoriesGrid;
