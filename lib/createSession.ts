import { NextApiRequest, NextApiResponse } from "next";
import getDb from "./db";
import { ExperimentGroup } from "../types/types";
import { v4 as uuidv4 } from "uuid";
import { assignExperimentGroup } from "./session";
import { getGuestCookie, setSessionCookie } from "./auth";

export async function createSession(
  req: NextApiRequest,
  res: NextApiResponse
): Promise<{ sessionId: string; group: ExperimentGroup }> {
  const db = await getDb();
  let group: ExperimentGroup = "control";
  const sessionId = getGuestCookie(req, res);

  // Admin kontrolü
  if (req.body.email === "admin@test.com" || req.body.password === "admin123") {
    group = "control";
    console.log("admin grubu = ", group);
  } else {
    group = assignExperimentGroup();
    console.log("user grubu = ", group);
  }

  await db.run(
    `INSERT INTO sessions (id, experiment_group, user_agent, ip)
     VALUES (?, ?, ?, ?)`,
    [
      sessionId,
      group,
      req.headers["user-agent"] || "",
      req.socket.remoteAddress || "",
    ]
  );

  setSessionCookie(req, res, sessionId);

  return { sessionId, group };
}
