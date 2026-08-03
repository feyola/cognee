declare module "d3-force-3d" {
  interface Force3D {
    (alpha: number): void;
    initialize?(nodes: unknown[], ...args: unknown[]): void;
    strength(value: number): Force3D;
    distanceMin(value: number): Force3D;
    distanceMax(value: number): Force3D;
  }

  export function forceCollide(radius?: number): Force3D;
  export function forceManyBody(): Force3D;
}
