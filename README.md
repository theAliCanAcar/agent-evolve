# Agent Evolve

Darwinian prompt engineering: genetic algorithms that mutate, crossover, and select their way to better answers.

TypeScript package `agent-evolve`.

## Install

```sh
npm install github:theAliCanAcar/agent-evolve#main
```

## How to use

```ts
import { evolve, express } from 'agent-evolve';

const result = await evolve({
  seedGenes: ['be concise', 'cite numbers', 'ask one clarifying question'],
  generations: 3,
  population: 6,
  fitness: async (prompt) => prompt.includes('concise') ? 0.8 : 0.2,
});
console.log(express(result.best.genes), result.history);
```

## Use cases

### 1. Improve a prompt with a fitness function

- **Scenario:** Hand-tuning instructions is slow.
- **Outcome:** evolve() mutates/crossovers genomes and keeps higher-scoring prompts.

### 2. Compare generations

- **Scenario:** You need evidence that a prompt got better.
- **Outcome:** Inspect GenerationStats and the winning expressed prompt.

## Dependencies

No package dependencies.

## Scripts

```sh
npm install
npm run build
```

## Notes

- Keep API keys in environment variables or a gitignored `.env`. Never commit secrets.
- Side-effecting tools should go through approval when used with `agent-approval`.
- Related packages live in separate repositories under [theAliCanAcar](https://github.com/theAliCanAcar).

## License

MIT
