import type { Tenant } from "./types";

/**
 * Open-source stub: local deployments do not provision cloud tenants.
 *
 * The OSS application always uses LocalProvider, which supplies the local
 * tenant directly. Keeping this no-op lets shared components type-check without
 * importing SaaS-only createTenant/getMyTenants modules that are not shipped in
 * the open-source frontend.
 */
export default async function resolveNewTenant(
  isCancelled: () => boolean,
): Promise<Tenant | null> {
  void isCancelled;
  return null;
}
