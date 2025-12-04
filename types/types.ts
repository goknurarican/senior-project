import { Dispatch, SetStateAction } from "react";

export type User = {
  succes: string;
  message: string;
  user: {
    name: string;
    email: string;
    id: string;
    role: string;
  };
};
export type SessionInfo = {
  sessionId: string | null;
  experimentGroup: ExperimentGroup;
};
export type ExperimentGroup =
  | "control"
  | "variant_a"
  | "variant_b"
  | "variant_c";
export type userCookie = {
  success: boolean;
  error: string;
  user: {
    id: string;
    email: string;
    name: string;
    role: string;
  };
};
export type CartState = {
  cartItems: CartItem[];
};

export type CartItem = {
  id: number;
  product_id: number;
  title: string;
  price: number;
  image: string;
  category: string;
  stock: number;
  quantity: number;
};
export type events = {
  id: number;
  session_id: string;
  event_type: string;
  page_url: string;
  timestamp: number;
  relative_t_ms: number;
  created_at: number;
};
export type experiments = {
  id: number;
  name: string;
  status: string;
  control_weight: number;
  variant_a_weight: number;
  variant_b_weight: number;
  created_at: number;
};
export type Product = {
  name: string;
  description: string;
  price: number;
  image: string;
  category: string;
};
export type products = {
  id: number;
  title: string;
  price: number;
  image: string;
  category: string;
  stock: number;
  description: string;
};
export type product_info = {
  product: products;
  quantity: number;
  setQuantity: Dispatch<SetStateAction<number>>;
  addToCart: () => Promise<void>;
};
export type scenario_triggers = {
  id: number;
  session_id: string;
  scenario_id: number;
  status: string;
  triggered_at: number;
};
export type scenarios = {
  id: number;
  name: string;
  type: string;
  target_page: string;
  selector: string;
  params: string;
  probability: number;
  enabled: number;
  created_at: number;
};
export type sessions = {
  id: string;
  experiment_group: string;
  created_at: number;
  user_agent: string;
  ip: string;
};
export type sqlite_sequence = {
  name: string;
  seq: string;
};
export type users = {
  id: string;
  email: string;
  password: string;
  role: string;
  name: string;
  created_at: number;
};
