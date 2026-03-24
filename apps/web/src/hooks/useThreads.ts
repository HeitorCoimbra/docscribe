'use client';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useSession } from 'next-auth/react';
import { getThreads, createThread, deleteThread } from '@/lib/api';

export function useThreads() {
  const { data: session } = useSession();
  const token = (session as { accessToken?: string })?.accessToken;

  return useQuery({
    queryKey: ['threads'],
    queryFn: () => getThreads(token!),
    enabled: !!token,
  });
}

export function useCreateThread() {
  const { data: session } = useSession();
  const token = (session as { accessToken?: string })?.accessToken;
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => createThread(token!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['threads'] });
    },
  });
}

export function useDeleteThread() {
  const { data: session } = useSession();
  const token = (session as { accessToken?: string })?.accessToken;
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => deleteThread(token!, id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['threads'] });
    },
  });
}
