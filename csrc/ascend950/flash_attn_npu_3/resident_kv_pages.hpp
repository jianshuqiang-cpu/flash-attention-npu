// L1 page assignment for the paired Q128, KV128, SWA (511, 0) prefill.
#ifndef FA950_RESIDENT_KV_PAGES_HPP
#define FA950_RESIDENT_KV_PAGES_HPP

#include <cstdint>

namespace fa950_resident {

constexpr uint32_t BANKS = 3U;
constexpr uint32_t WINDOW_PAGES = 5U;
constexpr uint32_t INVALID = 0xFFFFFFFFU;

struct Window {
    uint32_t banks[WINDOW_PAGES];
    uint32_t misses = 0U;
    bool lateTail = false;

    __aicore__ inline bool Miss(uint32_t page) const { return (misses >> page) & 1U; }
};

class PageBanks {
public:
    __aicore__ inline uint32_t Find(uint32_t page) const
    {
        for (uint32_t bank = 0; bank < BANKS; ++bank) {
            if (pages_[bank] == page) {
                return bank;
            }
        }
        return INVALID;
    }

    __aicore__ inline void Plan(Window& result, uint32_t begin, uint32_t end, uint32_t batch,
                                uint32_t head, uint32_t protectedBank)
    {
        if (batch_ != batch || head_ != head) {
            batch_ = batch;
            head_ = head;
            for (uint32_t bank = 0; bank < BANKS; ++bank) {
                pages_[bank] = INVALID;
            }
        }
        result.misses = 0U;
        result.lateTail = false;
        // Keep the first two pages of the next window after their current use.
        // Their offsets follow from full Q/KV tiles and a 511-token left window.
        const uint32_t nextFirst = end > 4U ? end - 4U : 0U;
        for (uint32_t page = begin; page < end; ++page) {
            uint32_t bank = Find(page);
            if (bank == INVALID) {
                const bool tail = page != begin && page + 1U == end;
                bank = Victim(page, end, protectedBank, nextFirst, tail);
                pages_[bank] = page;
                result.misses |= 1U << (page - begin);
                result.lateTail = tail && bank == protectedBank;
            }
            result.banks[page - begin] = bank;
            // A tail replacement can wait for its predecessor's final PV.
            // All other replacements must let that predecessor remain live.
            protectedBank = bank;
        }
    }

private:
    __aicore__ inline uint32_t Victim(uint32_t page, uint32_t end, uint32_t protectedBank,
                                     uint32_t keep, bool tail) const
    {
        uint32_t dead = INVALID;
        uint32_t farthest = INVALID;
        for (uint32_t bank = 0; bank < BANKS; ++bank) {
            if (bank == protectedBank) {
                continue;
            }
            const uint32_t cached = pages_[bank];
            if ((cached == keep || cached == keep + 1U) && cached < page) {
                continue;
            }
            if (cached == INVALID) {
                return bank;
            }
            if (cached < page || cached >= end) {
                if (dead == INVALID || cached < pages_[dead]) {
                    dead = bank;
                }
            } else if (farthest == INVALID || cached > pages_[farthest]) {
                farthest = bank;
            }
        }
        if (dead != INVALID) {
            return dead;
        }
        if (farthest != INVALID) {
            return farthest;
        }
        // The only fully pinned case is the tail of a five-page window.
        // Retire its predecessor before writing the tail into that bank.
        return tail ? protectedBank : INVALID;
    }

    uint32_t pages_[BANKS] = {INVALID, INVALID, INVALID};
    uint32_t batch_ = INVALID;
    uint32_t head_ = INVALID;
};

} // namespace fa950_resident

#endif
