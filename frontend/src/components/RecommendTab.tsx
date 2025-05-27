import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { apiService, RecommendRequest } from '../services/api';
import { Loader2, Target, Info, AlertCircle } from 'lucide-react';

const RecommendTab: React.FC = () => {
    const [userId, setUserId] = useState<string>('');
    const [benchmarkTags, setBenchmarkTags] = useState<string>('');
    const [numRecommend, setNumRecommend] = useState<string>('10');
    const [basicRecommendations, setBasicRecommendations] = useState<number[]>([]);
    const [advancedRecommendations, setAdvancedRecommendations] = useState<number[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string>('');
    const [noResultsMessage, setNoResultsMessage] = useState<string>('');

    // Example valid user IDs for demonstration
    const exampleUserIds = [115, 124, 2746, 5382, 8623, 8701, 12741, 13134, 24418, 24600];

    const handleRecommend = async (isAdvanced: boolean = false) => {
        try {
            setLoading(true);
            setError('');
            setNoResultsMessage('');

            const userIdNum = parseInt(userId.trim());
            const numRecommendNum = parseInt(numRecommend.trim());

            if (isNaN(userIdNum)) {
                throw new Error('请输入有效的用户ID');
            }

            if (isNaN(numRecommendNum) || numRecommendNum <= 0) {
                throw new Error('请输入有效的推荐数量');
            }

            const tags = benchmarkTags.trim()
                ? benchmarkTags.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
                : undefined;

            const request: RecommendRequest = {
                user_id: userIdNum,
                benchmark_tags: tags,
                num_recommend: numRecommendNum,
            };

            const response = isAdvanced
                ? await apiService.recommendAdvanced(request)
                : await apiService.recommend(request);

            if (response.question_ids.length === 0) {
                setNoResultsMessage('no_results');
            }

            if (isAdvanced) {
                setAdvancedRecommendations(response.question_ids);
            } else {
                setBasicRecommendations(response.question_ids);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : '推荐失败');
        } finally {
            setLoading(false);
        }
    };

    const fillExampleUserId = (id: number) => {
        setUserId(id.toString());
        setError('');
        setNoResultsMessage('');
    };

    const basicChartData = basicRecommendations.map((qid, index) => ({
        index: index + 1,
        questionId: qid,
    }));

    const advancedChartData = advancedRecommendations.map((qid, index) => ({
        index: index + 1,
        questionId: qid,
    }));

    return (
        <div className="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>题目推荐</CardTitle>
                    <CardDescription>
                        基于用户历史数据推荐合适的题目
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    {/* Info section with example user IDs */}
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                        <div className="flex items-center space-x-2 mb-2">
                            <Info className="h-4 w-4 text-blue-600" />
                            <span className="text-sm font-medium text-blue-800">提示</span>
                        </div>
                        <p className="text-sm text-blue-700 mb-2">
                            请使用数据集中存在的用户ID。以下是一些示例用户ID：
                        </p>
                        <div className="flex flex-wrap gap-1">
                            {exampleUserIds.slice(0, 8).map((id) => (
                                <button
                                    key={id}
                                    onClick={() => fillExampleUserId(id)}
                                    className="px-2 py-1 text-xs bg-blue-100 hover:bg-blue-200 text-blue-800 rounded transition-colors"
                                >
                                    {id}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="user-id">用户ID</Label>
                        <Input
                            id="user-id"
                            placeholder="例如: 115"
                            value={userId}
                            onChange={(e) => setUserId(e.target.value)}
                        />
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="benchmark-tags">基准标签 (可选，用逗号分隔)</Label>
                        <Input
                            id="benchmark-tags"
                            placeholder="例如: 1,2,3"
                            value={benchmarkTags}
                            onChange={(e) => setBenchmarkTags(e.target.value)}
                        />
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="num-recommend">推荐数量</Label>
                        <Input
                            id="num-recommend"
                            placeholder="例如: 10"
                            value={numRecommend}
                            onChange={(e) => setNumRecommend(e.target.value)}
                        />
                    </div>

                    {error && (
                        <div className="text-red-500 text-sm">{error}</div>
                    )}

                    <div className="flex gap-2">
                        <Button
                            onClick={() => handleRecommend(true)}
                            disabled={loading}
                            className="flex-1"
                        >
                            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                            <Target className="mr-2 h-4 w-4" />
                            推荐题目
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {advancedRecommendations.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle>高级推荐结果</CardTitle>
                        <CardDescription>
                            基于知识掌握度和题目难度的智能推荐
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-2">
                            {advancedRecommendations.map((qid, index) => (
                                <div key={index} className="text-center p-2 bg-muted rounded">
                                    <div className="text-sm text-muted-foreground">推荐 {index + 1}</div>
                                    <div className="font-semibold">题目 {qid}</div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}

            {noResultsMessage && (
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center space-x-2">
                            <AlertCircle className="h-5 w-5 text-orange-500" />
                            <span>未找到推荐结果</span>
                        </CardTitle>
                        <CardDescription>
                            用户ID {userId} 没有找到推荐结果
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
                            <p className="text-sm text-orange-800 mb-2">可能的原因：</p>
                            <ul className="text-sm text-orange-700 space-y-1 ml-4">
                                <li>• 该用户ID在数据集中不存在</li>
                                <li>• 该用户没有足够的历史数据</li>
                                <li>• 所有题目都已经被该用户尝试过</li>
                            </ul>
                        </div>

                        <div>
                            <p className="text-sm text-muted-foreground mb-2">请尝试使用以下示例用户ID：</p>
                            <div className="flex flex-wrap gap-2">
                                {exampleUserIds.slice(0, 5).map((id) => (
                                    <Button
                                        key={id}
                                        onClick={() => fillExampleUserId(id)}
                                        variant="outline"
                                        size="sm"
                                    >
                                        用户 {id}
                                    </Button>
                                ))}
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
};

export default RecommendTab; 