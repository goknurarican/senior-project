import { create } from "zustand";
type CartState = {
  cartItem: number;
  setCartItem: (value: number) => void;
};
export const useCartStore = create<CartState>((set) => ({
  cartItem: 0,
  setCartItem: (count: number) => set({ cartItem: count }),
}));
