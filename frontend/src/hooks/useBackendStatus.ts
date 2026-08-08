import { useCallback, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchStatus } from "@/api/model";
import { useAppStore } from "@/store/useAppStore";

/** Poll /api/status to drive the header GPU/provider state + online indicator. */
export function useBackendStatus(pollMs = 10000) {
  const set = useAppStore;

  const query = useQuery({
    queryKey: ["status"],
    queryFn: fetchStatus,
    refetchInterval: pollMs,
    retry: 1,
  });

  useEffect(() => {
    if (query.data) {
      set.getState().setProvider(query.data.provider);
      set.getState().setModelFamily(query.data.model_family);
      set.getState().setModel(query.data.model);
      set.getState().setMode(query.data.mode);
      set.getState().setGpu(query.data.gpu);
      set.getState().setBackendOnline(true);
    }
  }, [query.data, set]);

  useEffect(() => {
    if (query.isError) {
      set.getState().setBackendOnline(false);
    }
  }, [query.isError, set]);

  const refresh = useCallback(() => {
    void query.refetch();
  }, [query]);

  return { online: !query.isError, loading: query.isLoading, refresh };
}
