/**
 * agent-evolve: stop hand-writing prompts — grow them. A tiny genetic
 * algorithm over word (or character) sequences with fitness scoring,
 * mutation, crossover, tournament selection, and elitism.
 *
 *   const { champion } = await evolve({
 *     founders: [["please", "answer", "carefully"]],
 *     fitness: async (genes) => judge(genes.join(" ")),  // LLM-as-judge works
 *     generations: 8,
 *     seed: 42,
 *   });
 */

export type Genome = {
  id: string;
  genes: string[];
  fitness: number;
  generation: number;
  parents?: [string, string];
};

export type EvolveOptions = {
  /** Starting genomes (words or characters — evolve() doesn't care). */
  founders: string[][];
  /** Fitness function: higher is better. May call an LLM judge. */
  fitness: (genes: string[]) => number | Promise<number>;
  /** Generations to run, including the founder generation. Default 10. */
  generations?: number;
  /** Population size. Default 20. */
  populationSize?: number;
  /** Per-gene chance of mutation. Default 0.2. */
  mutationRate?: number;
  /** Chance a child is a crossover of two parents (else a clone). Default 0.7. */
  crossoverRate?: number;
  /** Top genomes copied unchanged each generation. Default 2. */
  elite?: number;
  /** Seed for reproducible evolution. Default 1337. */
  seed?: number;
  /** Vocabulary mutation may introduce. Defaults to a small demo lexicon. */
  vocabulary?: string[];
  /** Custom mutation operator (full control for fancy gene encodings). */
  mutator?: (genes: string[], rand: () => number) => string[];
};

export type GenerationStats = {
  generation: number;
  best: number;
  average: number;
  champion: string[];
};

export type EvolveResult = {
  champion: Genome;
  population: Genome[];
  history: GenerationStats[];
  generationsRun: number;
};

const DEMO_VOCABULARY = [
  "a",
  "the",
  "please",
  "think",
  "step",
  "by",
  "step",
  "be",
  "bold",
  "concise",
  "meticulous",
  "hack",
  "win",
  "ship",
];

/**
 * Run the evolutionary loop. Deterministic for a given seed and a
 * deterministic fitness function.
 */
export async function evolve(options: EvolveOptions): Promise<EvolveResult> {
  const rand = seededRandom(options.seed ?? 1337);
  const generations = Math.max(1, options.generations ?? 10);
  const populationSize = Math.max(2, options.populationSize ?? 20);
  const mutationRate = clamp01(options.mutationRate ?? 0.2);
  const crossoverRate = clamp01(options.crossoverRate ?? 0.7);
  const elite = Math.min(Math.max(0, options.elite ?? 2), populationSize - 1);
  const vocabulary = options.vocabulary ?? DEMO_VOCABULARY;
  const mutator =
    options.mutator ?? ((genes: string[], r: () => number) => mutateGenes(genes, r, mutationRate, vocabulary));

  if (options.founders.length === 0) {
    throw new Error("evolve() needs at least one founder genome.");
  }

  let counter = 0;
  const makeId = () => `genome-${++counter}`;

  async function spawn(genes: string[], generation: number, parents?: [string, string]): Promise<Genome> {
    return {
      id: makeId(),
      genes,
      fitness: await options.fitness(genes),
      generation,
      ...(parents ? { parents } : {}),
    };
  }

  async function evaluateAll(chromosomes: Array<{ genes: string[]; generation: number; parents?: [string, string] }>) {
    return Promise.all(chromosomes.map((c) => spawn(c.genes, c.generation, c.parents)));
  }

  // Founders, padded with mutants if the population is bigger than the litter.
  const founderChromosomes: Array<{ genes: string[]; generation: number }> = [];
  for (let i = 0; i < populationSize; i++) {
    const base = options.founders[i % options.founders.length];
    founderChromosomes.push({
      genes: i < options.founders.length ? [...base] : mutator([...base], rand),
      generation: 0,
    });
  }

  let population = await evaluateAll(founderChromosomes);
  const history: GenerationStats[] = [];

  for (let generation = 0; generation < generations; generation++) {
    population.sort((a, b) => b.fitness - a.fitness);
    history.push({
      generation,
      best: population[0].fitness,
      average: population.reduce((sum, g) => sum + g.fitness, 0) / population.length,
      champion: [...population[0].genes],
    });
    if (generation === generations - 1) break;

    const next: Array<{ genes: string[]; generation: number; parents?: [string, string] }> = [];
    for (let e = 0; e < elite; e++) {
      next.push({ genes: [...population[e].genes], generation: generation + 1 });
    }
    while (next.length < populationSize) {
      const a = tournamentSelect(population, { rand });
      const b = tournamentSelect(population, { rand });
      let genes =
        rand() < crossoverRate ? crossoverParents(a.genes, b.genes, rand)[0] : [...a.genes];
      genes = mutator(genes, rand);
      next.push({ genes, generation: generation + 1, parents: [a.id, b.id] });
    }
    population = await evaluateAll(next);
  }

  population.sort((a, b) => b.fitness - a.fitness);
  return {
    champion: population[0],
    population,
    history,
    generationsRun: generations,
  };
}

export type MutateOptions = {
  seed?: number;
  /** Per-gene mutation chance. Default 0.2. */
  rate?: number;
  vocabulary?: string[];
};

/** Mutate a gene sequence: swap, drop, duplicate, insert, or replace words. */
export function mutate(genes: string[], options: MutateOptions = {}): string[] {
  return mutateGenes(
    genes,
    seededRandom(options.seed ?? 1337),
    clamp01(options.rate ?? 0.2),
    options.vocabulary ?? DEMO_VOCABULARY
  );
}

/** Single-point crossover: each child gets the head of one parent, tail of the other. */
export function crossover(a: string[], b: string[], options: { seed?: number } = {}): [string[], string[]] {
  return crossoverParents(a, b, seededRandom(options.seed ?? 1337));
}

/** Pick the fittest of k random contenders (with replacement). */
export function tournamentSelect(
  pool: Genome[],
  options: { seed?: number; rand?: () => number; k?: number } = {}
): Genome {
  if (pool.length === 0) throw new Error("tournamentSelect() needs a non-empty pool.");
  const rand = options.rand ?? seededRandom(options.seed ?? 1337);
  const k = Math.max(1, options.k ?? 3);
  let champion = pool[Math.floor(rand() * pool.length)];
  for (let i = 1; i < k; i++) {
    const challenger = pool[Math.floor(rand() * pool.length)];
    if (challenger.fitness > champion.fitness) champion = challenger;
  }
  return champion;
}

/** Render a genome back into a prompt fragment. */
export function express(genes: string[]): string {
  return genes.join(" ");
}

function mutateGenes(genes: string[], rand: () => number, rate: number, vocabulary: string[]): string[] {
  const out = [...genes];
  for (let i = 0; i < out.length; i++) {
    if (rand() >= rate) continue;
    const op = rand();
    if (op < 0.2 && out.length > 1) {
      const j = Math.floor(rand() * out.length);
      [out[i], out[j]] = [out[j], out[i]]; // swap
    } else if (op < 0.4 && out.length > 1) {
      out.splice(i, 1); // drop
    } else if (op < 0.6) {
      out.splice(i, 0, out[i]); // duplicate
    } else if (op < 0.8) {
      out.splice(i, 0, pick(vocabulary, rand)); // insert
    } else {
      out[i] = pick(vocabulary, rand); // replace
    }
  }
  return out.length > 0 ? out : [...genes];
}

function crossoverParents(a: string[], b: string[], rand: () => number): [string[], string[]] {
  if (a.length < 2 || b.length < 2) return [[...a], [...b]];
  const pointA = 1 + Math.floor(rand() * (a.length - 1));
  const pointB = 1 + Math.floor(rand() * (b.length - 1));
  return [
    [...a.slice(0, pointA), ...b.slice(pointB)],
    [...b.slice(0, pointB), ...a.slice(pointA)],
  ];
}

function pick<T>(items: T[], rand: () => number): T {
  return items[Math.floor(rand() * items.length)];
}

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

/** Mulberry32 (same tiny PRNG as agent-chaos) so evolution is reproducible. */
function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}