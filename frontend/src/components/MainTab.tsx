import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { apiService, RecommendRequest, QuestionStats, PredictRequest } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Loader2, Target, Info, AlertCircle } from 'lucide-react';

const MainTab: React.FC = () => {
    const [userId, setUserId] = useState<string>('');
    const [benchmarkTags, setBenchmarkTags] = useState<string>('');
    const [numRecommend, setNumRecommend] = useState<string>('10');
    const [recommendations, setRecommendations] = useState<number[]>([]);
    const [questionStats, setQuestionStats] = useState<QuestionStats[]>([]);
    const [inputRtime, setInputRtime] = useState<string>('');
    const [inputCat, setInputCat] = useState<string>('');
    const [predictions, setPredictions] = useState<number[]>([]);
    const [loading, setLoading] = useState(false);
    const [predictLoading, setPredictLoading] = useState(false);
    const [error, setError] = useState<string>('');
    const [predictError, setPredictError] = useState<string>('');
    const [noResultsMessage, setNoResultsMessage] = useState<string>('');

    // Example valid user IDs for demonstration
    const exampleUserIds = [115, 124, 2746, 5382, 8623, 8701, 12741, 13134, 24418, 24600];

    // 推荐题目
    const handleRecommend = async () => {
        try {
            setLoading(true);
            setError('');
            setNoResultsMessage('');
            setRecommendations([]);
            setQuestionStats([]);
            setPredictions([]);
            setInputRtime('');
            setInputCat('');

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

            const response = await apiService.recommendAdvanced(request);
            if (response.question_ids.length === 0) {
                setNoResultsMessage('no_results');
                return;
            }
            setRecommendations(response.question_ids);

            // 获取题目信息
            const statsResponse = await apiService.getQuestionStatsByIds(response.question_ids);
            setQuestionStats(statsResponse);

            // 自动填写类别
            const cats = statsResponse.map(q => {
                return q.part ?? '0';
            });
            setInputCat(cats.join(','));
        } catch (err) {
            setError(err instanceof Error ? err.message : '推荐失败');
        } finally {
            setLoading(false);
        }
    };

    // 预测概率
    const handlePredict = async () => {
        try {
            setPredictLoading(true);
            setPredictError('');
            setPredictions([]);

            // Parse input strings to arrays
            const ids = recommendations;
            const rtime = inputRtime.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            const cat = inputCat.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

            if (ids.length === 0 || rtime.length === 0 || cat.length === 0) {
                throw new Error('请填写所有必需的输入');
            }
            if (ids.length !== rtime.length || ids.length !== cat.length) {
                throw new Error('题目ID、响应时间、类别长度必须一致');
            }

            const request: PredictRequest = {
                input_ids: ids,
                input_rtime: rtime,
                input_cat: cat,
            };
            const response = await apiService.predict(request);
            setPredictions(response.probs.slice(0, ids.length));
        } catch (err) {
            setPredictError(err instanceof Error ? err.message : '预测失败');
        } finally {
            setPredictLoading(false);
        }
    };

    const fillExampleUserId = (id: number) => {
        setUserId(id.toString());
        setError('');
        setNoResultsMessage('');
    };

    const chartData = predictions.map((prob, index) => ({
        index: index + 1,
        probability: prob,
    }));

    return (
        <div className="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>题目推荐与预测</CardTitle>
                    <CardDescription>
                        推荐题目后，自动填写题目ID和类别，手动填写响应时间，预测概率自动展示
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
                        <p className="text-sm text-muted-foreground">
                            基准标签：由使用的数据集提供，是一个或多个详细的标签代码，用于对问题进行聚类。标签的含义未被提供，但这些代码足以将类似的问题聚类在一起。
                        </p>
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
                            onClick={handleRecommend}
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

            {/* 推荐题目结果 */}
            {recommendations.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle>推荐题目结果</CardTitle>
                        <CardDescription>
                            根据用户历史数据和基准标签推荐的题目列表
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {/* 推荐题目信息 */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                            {recommendations.map((qid, index) => {
                                const stat = questionStats.find(q => q.question_id === qid);
                                return (
                                    <div key={index} className="text-left p-2 bg-muted rounded border">
                                        <div className="text-sm text-muted-foreground mb-1">推荐 {index + 1}</div>
                                        <div className="font-semibold mb-1">题目 {qid}</div>
                                        <div className="flex flex-wrap gap-1 mb-1">
                                            <span className="inline-block px-2 py-0.5 text-xs rounded bg-orange-100 text-orange-800">
                                                难度: {stat?.difficulty !== undefined && stat?.difficulty !== null ? stat.difficulty.toFixed(3) : '-'}
                                            </span>
                                            <span className="inline-block px-2 py-0.5 text-xs rounded bg-green-100 text-green-800">
                                                区分度: {stat?.discrimination !== undefined && stat?.discrimination !== null ? stat.discrimination.toFixed(3) : '-'}
                                            </span>
                                            <span className="inline-block px-2 py-0.5 text-xs rounded bg-yellow-100 text-yellow-800">
                                                标签: {stat?.tags ?? '-'}
                                            </span>
                                            <span className="inline-block px-2 py-0.5 text-xs rounded bg-pink-100 text-pink-800">
                                                TOEIC 章节: {stat?.part ?? '-'}
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* 预测输入 */}
            {recommendations.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle>预测输入</CardTitle>
                        <CardDescription>
                            题目ID和类别已自动填写，请输入响应时间（用逗号分隔，数量需与题目数一致）
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="input-ids">题目ID序列</Label>
                            <Input
                                id="input-ids"
                                value={recommendations.join(',')}
                                readOnly
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="input-cat">题目类别序列</Label>
                            <Input
                                id="input-cat"
                                value={inputCat}
                                readOnly
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="input-rtime">响应时间序列 (用逗号分隔)</Label>
                            <Input
                                id="input-rtime"
                                placeholder={`如: ${Array(recommendations.length).fill(1000).join(',')}`}
                                value={inputRtime}
                                onChange={(e) => setInputRtime(e.target.value)}
                            />
                        </div>
                        {predictError && (
                            <div className="text-red-500 text-sm">{predictError}</div>
                        )}
                        <Button onClick={handlePredict} disabled={predictLoading} className="w-full">
                            {predictLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                            {predictLoading ? '预测中...' : '开始预测'}
                        </Button>
                    </CardContent>
                </Card>
            )}

            {/* 预测结果 */}
            {predictions.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle>预测结果</CardTitle>
                        <CardDescription>
                            模型对每个题目的预测概率
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="h-64 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="index" />
                                    <YAxis domain={[0, 1]} />
                                    <Tooltip
                                        formatter={(value: number) => [value.toFixed(4), '预测概率']}
                                        labelFormatter={(label) => `题目 ${label}`}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="probability"
                                        stroke="#8884d8"
                                        strokeWidth={2}
                                        dot={{ r: 4 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
                            {predictions.map((prob, index) => (
                                <div key={index} className="text-center p-2 bg-muted rounded">
                                    <div className="text-sm text-muted-foreground">题目 {index + 1}</div>
                                    <div className="font-semibold">{prob.toFixed(4)}</div>
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

export default MainTab; 