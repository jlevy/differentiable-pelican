/* tslint:disable */
/* eslint-disable */

/**
 * The main optimizer exposed to JavaScript via wasm-bindgen.
 */
export class PelicanOptimizer {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the current step number.
     */
    current_step(): number;
    /**
     * Render the current model state as grayscale u8 pixels.
     */
    get_rendered_pixels(width: number, height: number): Uint8Array;
    /**
     * Get SVG representation of the current model.
     */
    get_svg(width: number, height: number): string;
    /**
     * Check if optimization is complete.
     */
    is_done(): boolean;
    /**
     * Create a new optimizer from a target image (grayscale u8 pixels, row-major).
     */
    constructor(target_pixels: Uint8Array, width: number, height: number, total_steps: number);
    /**
     * Run one optimization step. Returns the loss value.
     */
    step(): number;
    /**
     * Run multiple optimization steps. Returns array of loss values.
     */
    step_n(n: number): Float32Array;
    /**
     * Get total steps.
     */
    total_steps(): number;
}

/**
 * Get SVG of the initial pelican.
 */
export function initial_pelican_svg(width: number, height: number): string;

/**
 * Render the initial pelican as grayscale u8 pixels (no optimization).
 */
export function render_initial_pelican(width: number, height: number): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_pelicanoptimizer_free: (a: number, b: number) => void;
    readonly initial_pelican_svg: (a: number, b: number) => [number, number];
    readonly pelicanoptimizer_current_step: (a: number) => number;
    readonly pelicanoptimizer_get_rendered_pixels: (a: number, b: number, c: number) => [number, number];
    readonly pelicanoptimizer_get_svg: (a: number, b: number, c: number) => [number, number];
    readonly pelicanoptimizer_is_done: (a: number) => number;
    readonly pelicanoptimizer_new: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly pelicanoptimizer_step: (a: number) => number;
    readonly pelicanoptimizer_step_n: (a: number, b: number) => [number, number];
    readonly pelicanoptimizer_total_steps: (a: number) => number;
    readonly render_initial_pelican: (a: number, b: number) => [number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
