// pages/api/scenarios/active.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import getDb from '../../../lib/db';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { page, group } = req.query;
  const db = await getDb();
  
  // Get ALL scenarios but we'll filter on client side based on enabled status
  // This ensures changes are reflected immediately
  const scenarios = await db.all(`
    SELECT * FROM scenarios 
    WHERE (target_page = ? OR target_page = "*")
    ORDER BY probability DESC
  `, [page || '/']);
  
  // Adjust probabilities based on experiment group
  const adjustedScenarios = scenarios.map(scenario => {
    let adjustedProbability = scenario.probability;
    
    // Control group gets no scenarios
    if (group === 'control') {
      adjustedProbability = 0;
    }
    // Variant A gets lower probability (30% tier)
    else if (group === 'variant_a') {
      adjustedProbability = scenario.probability * 0.5;
    }
    // Variant B gets medium probability (60% tier) 
    else if (group === 'variant_b') {
      adjustedProbability = scenario.probability * 0.8;
    }
    
    return {
      ...scenario,
      probability: adjustedProbability
    };
  });
  
  res.status(200).json(adjustedScenarios);
}
